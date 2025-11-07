#
# Plan:
# 1. Create comprehensive rate limiting system for API protection
# 2. Support multiple rate limiting algorithms (token bucket, sliding window)
# 3. Per-user, per-IP, and global rate limiting
# 4. Different limits for different endpoints and user roles
# 5. Redis backend support for distributed rate limiting
# 6. Rate limit headers and proper HTTP responses
# 7. Configurable time windows and burst allowances
#

"""
Rate Limiter for SereneSense API
Protects API from abuse and ensures fair resource usage.

Features:
- Multiple rate limiting algorithms
- Per-user and per-IP limiting
- Role-based rate limits
- Endpoint-specific limits
- Redis backend support
- Configurable time windows
"""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import hashlib
import json
from datetime import datetime, timedelta

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from core.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""

    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    burst: Optional[int] = None  # Burst allowance (for token bucket)


@dataclass
class RateLimitResult:
    """Rate limit check result"""

    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None


class TokenBucket:
    """
    Token bucket rate limiting algorithm.
    Allows burst traffic up to bucket capacity.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def is_allowed(self, tokens_requested: int = 1) -> Tuple[bool, int]:
        """
        Check if request is allowed and consume tokens.

        Args:
            tokens_requested: Number of tokens to consume

        Returns:
            Tuple of (allowed, remaining_tokens)
        """
        now = time.time()

        # Refill tokens based on elapsed time
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

        # Check if enough tokens available
        if self.tokens >= tokens_requested:
            self.tokens -= tokens_requested
            return True, int(self.tokens)
        else:
            return False, int(self.tokens)

    def time_until_available(self, tokens_needed: int = 1) -> float:
        """
        Calculate time until tokens are available.

        Args:
            tokens_needed: Number of tokens needed

        Returns:
            Time in seconds until tokens available
        """
        if tokens_needed <= self.tokens:
            return 0.0

        tokens_deficit = tokens_needed - self.tokens
        return tokens_deficit / self.refill_rate


class SlidingWindowCounter:
    """
    Sliding window rate limiting algorithm.
    Provides precise rate limiting over a sliding time window.
    """

    def __init__(self, limit: int, window_seconds: int):
        """
        Initialize sliding window counter.

        Args:
            limit: Request limit per window
            window_seconds: Window duration in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = []  # List of request timestamps

    def is_allowed(self) -> Tuple[bool, int, int]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests if req_time > window_start]

        # Check if limit exceeded
        if len(self.requests) < self.limit:
            self.requests.append(now)
            remaining = self.limit - len(self.requests)
            reset_time = int(now + self.window_seconds)
            return True, remaining, reset_time
        else:
            remaining = 0
            # Reset time is when the oldest request expires
            reset_time = int(self.requests[0] + self.window_seconds)
            return False, remaining, reset_time


class MemoryRateLimiter:
    """
    In-memory rate limiter using various algorithms.
    Suitable for single-instance deployments.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory rate limiter.

        Args:
            config: Rate limiter configuration
        """
        self.config = config
        self.algorithm = config.get("algorithm", "sliding_window")  # token_bucket, sliding_window

        # Storage for different limit types
        self.user_limiters = {}
        self.ip_limiters = {}
        self.global_limiter = None

        # Default limits
        self.default_limits = {
            "requests_per_minute": config.get("requests_per_minute", 60),
            "requests_per_hour": config.get("requests_per_hour", 1000),
            "burst_size": config.get("burst_size", 10),
        }

        # Role-based limits
        self.role_limits = config.get(
            "role_limits",
            {
                "user": {"requests_per_minute": 60, "requests_per_hour": 1000},
                "operator": {"requests_per_minute": 120, "requests_per_hour": 2000},
                "admin": {"requests_per_minute": 300, "requests_per_hour": 5000},
            },
        )

        # Endpoint-specific limits
        self.endpoint_limits = config.get(
            "endpoint_limits",
            {
                "/detect": {"requests_per_minute": 30},
                "/batch": {"requests_per_minute": 10},
                "/ws/realtime": {"connections": 5},
            },
        )

        # Initialize global limiter
        if config.get("global_limit_enabled", True):
            global_limit = config.get("global_requests_per_minute", 1000)
            self.global_limiter = self._create_limiter(global_limit, 60)

        logger.info(f"Memory rate limiter initialized with {self.algorithm} algorithm")

    def _create_limiter(self, requests: int, window_seconds: int, burst: int = None):
        """Create rate limiter based on algorithm"""
        if self.algorithm == "token_bucket":
            capacity = burst or requests
            refill_rate = requests / window_seconds
            return TokenBucket(capacity, refill_rate)

        else:  # sliding_window
            return SlidingWindowCounter(requests, window_seconds)

    def _get_limit_for_user(self, user_info: Dict[str, Any]) -> Dict[str, int]:
        """Get rate limits for user based on roles"""
        user_roles = user_info.get("roles", ["user"])

        # Find the highest role limits
        max_limits = self.default_limits.copy()

        for role in user_roles:
            if role in self.role_limits:
                role_limits = self.role_limits[role]
                for key, value in role_limits.items():
                    max_limits[key] = max(max_limits.get(key, 0), value)

        return max_limits

    def is_allowed(
        self,
        identifier: str,
        limit_type: str = "user",
        user_info: Dict[str, Any] = None,
        endpoint: str = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            identifier: User ID, IP address, or other identifier
            limit_type: Type of limit ('user', 'ip', 'global')
            user_info: User information for role-based limits
            endpoint: Endpoint for endpoint-specific limits

        Returns:
            Rate limit check result
        """
        # Check global limits first
        if self.global_limiter:
            if self.algorithm == "token_bucket":
                allowed, remaining = self.global_limiter.is_allowed()
                if not allowed:
                    retry_after = int(self.global_limiter.time_until_available())
                    return RateLimitResult(False, 0, int(time.time() + retry_after), retry_after)
            else:
                allowed, remaining, reset_time = self.global_limiter.is_allowed()
                if not allowed:
                    retry_after = reset_time - int(time.time())
                    return RateLimitResult(False, remaining, reset_time, retry_after)

        # Get appropriate limiter storage
        if limit_type == "user":
            limiters = self.user_limiters
        elif limit_type == "ip":
            limiters = self.ip_limiters
        else:
            # For other types, use user limiters
            limiters = self.user_limiters

        # Create limiter key
        limiter_key = f"{identifier}:{endpoint}" if endpoint else identifier

        # Get or create rate limiter for this identifier
        if limiter_key not in limiters:
            # Determine limits
            if endpoint and endpoint in self.endpoint_limits:
                endpoint_limit = self.endpoint_limits[endpoint]
                requests = endpoint_limit.get(
                    "requests_per_minute", self.default_limits["requests_per_minute"]
                )
            elif user_info:
                user_limits = self._get_limit_for_user(user_info)
                requests = user_limits["requests_per_minute"]
            else:
                requests = self.default_limits["requests_per_minute"]

            # Create limiter
            limiters[limiter_key] = self._create_limiter(requests, 60)

        limiter = limiters[limiter_key]

        # Check rate limit
        if self.algorithm == "token_bucket":
            allowed, remaining = limiter.is_allowed()
            reset_time = int(time.time() + 60)  # Approximate reset time

            if not allowed:
                retry_after = int(limiter.time_until_available())
                return RateLimitResult(False, remaining, reset_time, retry_after)
            else:
                return RateLimitResult(True, remaining, reset_time)

        else:  # sliding_window
            allowed, remaining, reset_time = limiter.is_allowed()

            if not allowed:
                retry_after = reset_time - int(time.time())
                return RateLimitResult(False, remaining, reset_time, retry_after)
            else:
                return RateLimitResult(True, remaining, reset_time)

    def reset_limits(self, identifier: str, limit_type: str = "user"):
        """Reset rate limits for identifier"""
        if limit_type == "user" and identifier in self.user_limiters:
            del self.user_limiters[identifier]
        elif limit_type == "ip" and identifier in self.ip_limiters:
            del self.ip_limiters[identifier]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "algorithm": self.algorithm,
            "user_limiters": len(self.user_limiters),
            "ip_limiters": len(self.ip_limiters),
            "global_limiter_active": self.global_limiter is not None,
            "default_limits": self.default_limits,
            "role_limits": self.role_limits,
            "endpoint_limits": self.endpoint_limits,
        }


class RedisRateLimiter:
    """
    Redis-based rate limiter for distributed deployments.
    Uses Redis for shared state across multiple instances.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Redis rate limiter.

        Args:
            config: Rate limiter configuration
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.config = config
        redis_config = config.get("redis", {})

        # Connect to Redis
        self.redis = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            decode_responses=True,
        )

        # Test connection
        try:
            self.redis.ping()
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Key prefix for rate limiting keys
        self.key_prefix = config.get("key_prefix", "serenesense:ratelimit:")

        # Default limits (same as memory limiter)
        self.default_limits = {
            "requests_per_minute": config.get("requests_per_minute", 60),
            "requests_per_hour": config.get("requests_per_hour", 1000),
        }

        self.role_limits = config.get(
            "role_limits",
            {
                "user": {"requests_per_minute": 60},
                "operator": {"requests_per_minute": 120},
                "admin": {"requests_per_minute": 300},
            },
        )

        self.endpoint_limits = config.get("endpoint_limits", {})

    def _get_redis_key(self, identifier: str, window: str, endpoint: str = None) -> str:
        """Generate Redis key for rate limiting"""
        key_parts = [self.key_prefix, identifier, window]
        if endpoint:
            # Hash endpoint to avoid key length issues
            endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
            key_parts.append(endpoint_hash)

        return ":".join(key_parts)

    def is_allowed(
        self,
        identifier: str,
        limit_type: str = "user",
        user_info: Dict[str, Any] = None,
        endpoint: str = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed using Redis.

        Args:
            identifier: User ID, IP address, or other identifier
            limit_type: Type of limit
            user_info: User information
            endpoint: Endpoint path

        Returns:
            Rate limit check result
        """
        # Determine request limit
        if endpoint and endpoint in self.endpoint_limits:
            requests_per_minute = self.endpoint_limits[endpoint].get("requests_per_minute", 60)
        elif user_info:
            user_roles = user_info.get("roles", ["user"])
            requests_per_minute = self.default_limits["requests_per_minute"]

            for role in user_roles:
                if role in self.role_limits:
                    role_limit = self.role_limits[role].get("requests_per_minute", 0)
                    requests_per_minute = max(requests_per_minute, role_limit)
        else:
            requests_per_minute = self.default_limits["requests_per_minute"]

        # Use sliding window approach with Redis
        now = int(time.time())
        window_start = now - 60  # 1 minute window

        # Redis key for this identifier and window
        redis_key = self._get_redis_key(identifier, "minute", endpoint)

        # Lua script for atomic sliding window check
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- Count current requests
        local current = redis.call('ZCARD', key)
        
        if current < limit then
            -- Add new request
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, 60)
            return {1, limit - current - 1, now + 60}
        else
            -- Get oldest request for reset time calculation
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local reset_time = now + 60
            if #oldest > 0 then
                reset_time = tonumber(oldest[2]) + 60
            end
            return {0, 0, reset_time}
        end
        """

        try:
            # Execute Lua script
            result = self.redis.eval(
                lua_script, 1, redis_key, window_start, now, requests_per_minute
            )

            allowed = bool(result[0])
            remaining = int(result[1])
            reset_time = int(result[2])

            retry_after = None
            if not allowed:
                retry_after = max(1, reset_time - now)

            return RateLimitResult(allowed, remaining, reset_time, retry_after)

        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fallback to allowing request if Redis fails
            return RateLimitResult(True, requests_per_minute, now + 60)

    def reset_limits(self, identifier: str, limit_type: str = "user"):
        """Reset rate limits for identifier"""
        try:
            pattern = f"{self.key_prefix}{identifier}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Error resetting rate limits: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        try:
            # Count active rate limit keys
            pattern = f"{self.key_prefix}*"
            active_keys = len(self.redis.keys(pattern))

            return {
                "backend": "redis",
                "active_keys": active_keys,
                "default_limits": self.default_limits,
                "role_limits": self.role_limits,
                "endpoint_limits": self.endpoint_limits,
                "redis_info": {
                    "connected": True,
                    "memory_usage": self.redis.info().get("used_memory_human", "unknown"),
                },
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {"backend": "redis", "error": str(e)}


class RateLimiter:
    """
    Main rate limiter class that coordinates different backends.
    Automatically chooses between memory and Redis based on configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rate limiter.

        Args:
            config: Rate limiter configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)

        if not self.enabled:
            logger.info("Rate limiting disabled")
            return

        # Choose backend
        backend = config.get("backend", "memory")

        if backend == "redis" and REDIS_AVAILABLE:
            try:
                self.limiter = RedisRateLimiter(config)
                logger.info("Using Redis rate limiter backend")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis backend, falling back to memory: {e}")
                self.limiter = MemoryRateLimiter(config)
        else:
            self.limiter = MemoryRateLimiter(config)
            logger.info("Using memory rate limiter backend")

    def is_allowed(
        self,
        identifier: str,
        limit_type: str = "user",
        user_info: Dict[str, Any] = None,
        endpoint: str = None,
    ) -> bool:
        """
        Check if request is allowed.

        Args:
            identifier: User ID, IP address, or other identifier
            limit_type: Type of limit ('user', 'ip')
            user_info: User information for role-based limits
            endpoint: Endpoint for endpoint-specific limits

        Returns:
            True if request is allowed
        """
        if not self.enabled:
            return True

        try:
            result = self.limiter.is_allowed(identifier, limit_type, user_info, endpoint)
            return result.allowed
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Allow request if rate limiter fails
            return True

    def get_rate_limit_info(
        self,
        identifier: str,
        limit_type: str = "user",
        user_info: Dict[str, Any] = None,
        endpoint: str = None,
    ) -> RateLimitResult:
        """
        Get detailed rate limit information.

        Args:
            identifier: User ID, IP address, or other identifier
            limit_type: Type of limit
            user_info: User information
            endpoint: Endpoint path

        Returns:
            Rate limit result with headers info
        """
        if not self.enabled:
            return RateLimitResult(True, 1000, int(time.time() + 3600))

        try:
            return self.limiter.is_allowed(identifier, limit_type, user_info, endpoint)
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return RateLimitResult(True, 1000, int(time.time() + 3600))

    def reset_limits(self, identifier: str, limit_type: str = "user"):
        """Reset rate limits for identifier"""
        if self.enabled:
            try:
                self.limiter.reset_limits(identifier, limit_type)
            except Exception as e:
                logger.error(f"Error resetting rate limits: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        if not self.enabled:
            return {"enabled": False}

        try:
            stats = self.limiter.get_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting rate limiter stats: {e}")
            return {"enabled": True, "error": str(e)}


def create_rate_limiter(config_path: str = None) -> RateLimiter:
    """
    Create rate limiter from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured rate limiter
    """
    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        rate_limit_config = config_dict.get("rate_limiting", {})
    else:
        rate_limit_config = {"enabled": False}

    return RateLimiter(rate_limit_config)


if __name__ == "__main__":
    # Demo: Rate limiting system
    import argparse

    parser = argparse.ArgumentParser(description="Rate Limiter Demo")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--backend", default="memory", choices=["memory", "redis"], help="Rate limiter backend"
    )
    parser.add_argument("--test-requests", type=int, default=10, help="Number of test requests")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create rate limiter
        if args.config:
            rate_limiter = create_rate_limiter(args.config)
        else:
            config = {
                "enabled": True,
                "backend": args.backend,
                "requests_per_minute": 5,  # Low limit for testing
                "algorithm": "sliding_window",
            }
            rate_limiter = RateLimiter(config)

        if not rate_limiter.enabled:
            print("❌ Rate limiting is disabled")
        else:
            print(f"🔒 Testing rate limiter with {args.backend} backend")

            # Test rate limiting
            user_id = "test_user"
            user_info = {"roles": ["user"]}

            for i in range(args.test_requests):
                result = rate_limiter.get_rate_limit_info(user_id, "user", user_info)

                if result.allowed:
                    print(f"✅ Request {i+1}: Allowed (remaining: {result.remaining})")
                else:
                    print(f"❌ Request {i+1}: Blocked (retry after: {result.retry_after}s)")

                # Small delay between requests
                time.sleep(0.1)

            # Show stats
            stats = rate_limiter.get_stats()
            print(f"\n📊 Rate Limiter Stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Rate limiter demo failed: {e}")
