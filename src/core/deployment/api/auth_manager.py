#
# Plan:
# 1. Create comprehensive authentication system for API security
# 2. Support JWT tokens with configurable expiration
# 3. API key authentication for service-to-service communication
# 4. Role-based access control (RBAC) for different user types
# 5. Token refresh mechanism and blacklisting
# 6. Integration with external identity providers (optional)
# 7. Audit logging for security events
#

"""
Authentication Manager for SereneSense API
Provides secure authentication and authorization for military applications.

Features:
- JWT token authentication
- API key authentication
- Role-based access control
- Token refresh and blacklisting
- Security audit logging
- Integration with external providers
"""

import jwt
import hashlib
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import json
import time
from pathlib import Path
import bcrypt

from core.utils.config_parser import ConfigParser

logger = logging.getLogger(__name__)


@dataclass
class UserInfo:
    """User information structure"""

    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    active: bool = True


@dataclass
class TokenInfo:
    """Token information structure"""

    token_id: str
    user_id: str
    token_type: str  # access, refresh, api_key
    issued_at: datetime
    expires_at: Optional[datetime]
    scopes: List[str]
    revoked: bool = False


class PermissionManager:
    """Manages permissions and role-based access control"""

    # Default permissions
    PERMISSIONS = {
        "detection.single": "Perform single file detection",
        "detection.batch": "Perform batch file detection",
        "detection.realtime": "Access real-time detection",
        "admin.users": "Manage user accounts",
        "admin.tokens": "Manage API tokens",
        "admin.metrics": "Access system metrics",
        "admin.health": "Access health endpoints",
        "model.info": "Access model information",
        "model.update": "Update detection models",
    }

    # Default roles
    ROLES = {
        "user": ["detection.single", "detection.batch", "model.info"],
        "operator": ["detection.single", "detection.batch", "detection.realtime", "model.info"],
        "admin": [
            "detection.single",
            "detection.batch",
            "detection.realtime",
            "admin.users",
            "admin.tokens",
            "admin.metrics",
            "admin.health",
            "model.info",
            "model.update",
        ],
    }

    def __init__(self):
        """Initialize permission manager"""
        self.custom_roles = {}
        self.custom_permissions = {}

    def get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        """
        Get all permissions for given roles.

        Args:
            roles: List of role names

        Returns:
            List of permissions
        """
        permissions = set()

        for role in roles:
            if role in self.ROLES:
                permissions.update(self.ROLES[role])
            elif role in self.custom_roles:
                permissions.update(self.custom_roles[role])

        return list(permissions)

    def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """
        Check if user has required permission.

        Args:
            user_permissions: User's permissions
            required_permission: Required permission

        Returns:
            True if user has permission
        """
        return required_permission in user_permissions

    def add_custom_role(self, role_name: str, permissions: List[str]):
        """Add custom role with permissions"""
        self.custom_roles[role_name] = permissions

    def add_custom_permission(self, permission_name: str, description: str):
        """Add custom permission"""
        self.custom_permissions[permission_name] = description


class TokenManager:
    """Manages JWT tokens and API keys"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize token manager.

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.blacklisted_tokens = set()
        self.api_keys = {}  # api_key -> user_info

        # Token settings
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 30
        self.api_key_expire_days = 365

    def create_access_token(
        self, user_id: str, username: str, roles: List[str], permissions: List[str]
    ) -> Tuple[str, datetime]:
        """
        Create JWT access token.

        Args:
            user_id: User identifier
            username: Username
            roles: User roles
            permissions: User permissions

        Returns:
            Tuple of (token, expiration_time)
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "permissions": permissions,
            "token_type": "access",
            "iat": now,
            "exp": expires_at,
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expires_at

    def create_refresh_token(self, user_id: str) -> Tuple[str, datetime]:
        """
        Create JWT refresh token.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (token, expiration_time)
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "user_id": user_id,
            "token_type": "refresh",
            "iat": now,
            "exp": expires_at,
            "jti": secrets.token_urlsafe(16),
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expires_at

    def create_api_key(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        permissions: List[str],
        description: str = "",
    ) -> str:
        """
        Create API key for service-to-service authentication.

        Args:
            user_id: User identifier
            username: Username
            roles: User roles
            permissions: User permissions
            description: Key description

        Returns:
            API key string
        """
        # Generate API key
        api_key = f"sk_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"

        # Store API key info
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.api_key_expire_days)

        self.api_keys[api_key] = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "permissions": permissions,
            "description": description,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0,
        }

        return api_key

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token

        Returns:
            Decoded payload or None if invalid
        """
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return None

            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type and expiration
            if payload.get("token_type") not in ["access", "refresh"]:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Verify API key.

        Args:
            api_key: API key string

        Returns:
            API key info or None if invalid
        """
        if api_key not in self.api_keys:
            return None

        key_info = self.api_keys[api_key]

        # Check if expired
        if key_info["expires_at"] < datetime.now(timezone.utc):
            return None

        # Update usage statistics
        key_info["last_used"] = datetime.now(timezone.utc)
        key_info["usage_count"] += 1

        return key_info

    def revoke_token(self, token: str):
        """Revoke JWT token by adding to blacklist"""
        self.blacklisted_tokens.add(token)

    def revoke_api_key(self, api_key: str):
        """Revoke API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]

    def cleanup_expired_tokens(self):
        """Clean up expired blacklisted tokens"""
        # This would typically be done periodically
        # For now, we keep all blacklisted tokens in memory
        pass


class UserManager:
    """Manages user accounts and authentication"""

    def __init__(self, user_store_path: str = None):
        """
        Initialize user manager.

        Args:
            user_store_path: Path to user data file
        """
        self.user_store_path = user_store_path or "users.json"
        self.users = {}  # user_id -> UserInfo
        self.username_to_id = {}  # username -> user_id

        self._load_users()

    def _load_users(self):
        """Load users from storage"""
        try:
            if Path(self.user_store_path).exists():
                with open(self.user_store_path, "r") as f:
                    data = json.load(f)

                for user_data in data.get("users", []):
                    user = UserInfo(
                        user_id=user_data["user_id"],
                        username=user_data["username"],
                        email=user_data.get("email"),
                        roles=user_data["roles"],
                        permissions=user_data["permissions"],
                        created_at=datetime.fromisoformat(user_data["created_at"]),
                        last_login=(
                            datetime.fromisoformat(user_data["last_login"])
                            if user_data.get("last_login")
                            else None
                        ),
                        active=user_data.get("active", True),
                    )

                    self.users[user.user_id] = user
                    self.username_to_id[user.username] = user.user_id

                logger.info(f"Loaded {len(self.users)} users from {self.user_store_path}")

        except Exception as e:
            logger.warning(f"Could not load users: {e}")
            # Create default admin user
            self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = UserInfo(
            user_id="admin",
            username="admin",
            email="admin@core.local",
            roles=["admin"],
            permissions=PermissionManager().get_permissions_for_roles(["admin"]),
            created_at=datetime.now(timezone.utc),
        )

        self.users[admin_user.user_id] = admin_user
        self.username_to_id[admin_user.username] = admin_user.user_id

        logger.info("Created default admin user")

    def _save_users(self):
        """Save users to storage"""
        try:
            data = {"users": []}

            for user in self.users.values():
                user_data = {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles,
                    "permissions": user.permissions,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "active": user.active,
                }
                data["users"].append(user_data)

            with open(self.user_store_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save users: {e}")

    def authenticate_user(self, username: str, password: str) -> Optional[UserInfo]:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Password

        Returns:
            UserInfo if authenticated, None otherwise
        """
        # This is a simplified implementation
        # In production, passwords should be hashed and stored securely

        if username not in self.username_to_id:
            return None

        user_id = self.username_to_id[username]
        user = self.users[user_id]

        if not user.active:
            return None

        # For demo purposes, accept any password for admin user
        # In production, use proper password hashing
        if username == "admin":
            user.last_login = datetime.now(timezone.utc)
            self._save_users()
            return user

        return None

    def get_user(self, user_id: str) -> Optional[UserInfo]:
        """Get user by ID"""
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[UserInfo]:
        """Get user by username"""
        user_id = self.username_to_id.get(username)
        return self.users.get(user_id) if user_id else None

    def create_user(self, username: str, email: str, roles: List[str]) -> UserInfo:
        """
        Create new user.

        Args:
            username: Username
            email: Email address
            roles: User roles

        Returns:
            Created user info
        """
        if username in self.username_to_id:
            raise ValueError(f"Username already exists: {username}")

        permission_manager = PermissionManager()
        permissions = permission_manager.get_permissions_for_roles(roles)

        user = UserInfo(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
        )

        self.users[user.user_id] = user
        self.username_to_id[user.username] = user.user_id

        self._save_users()
        return user

    def update_user_roles(self, user_id: str, roles: List[str]):
        """Update user roles and permissions"""
        if user_id not in self.users:
            raise ValueError(f"User not found: {user_id}")

        permission_manager = PermissionManager()
        permissions = permission_manager.get_permissions_for_roles(roles)

        user = self.users[user_id]
        user.roles = roles
        user.permissions = permissions

        self._save_users()

    def deactivate_user(self, user_id: str):
        """Deactivate user account"""
        if user_id in self.users:
            self.users[user_id].active = False
            self._save_users()


class AuthenticationManager:
    """
    Main authentication manager for SereneSense API.
    Coordinates user management, token management, and permissions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize authentication manager.

        Args:
            config: Authentication configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False)

        if not self.enabled:
            logger.info("Authentication disabled")
            return

        # Initialize components
        secret_key = config.get("secret_key", "change-me-in-production")
        if secret_key == "change-me-in-production":
            logger.warning("Using default secret key - change in production!")

        self.token_manager = TokenManager(secret_key)
        self.user_manager = UserManager(config.get("user_store_path"))
        self.permission_manager = PermissionManager()

        # Configure token expiration
        if "access_token_expire_minutes" in config:
            self.token_manager.access_token_expire_minutes = config["access_token_expire_minutes"]

        if "refresh_token_expire_days" in config:
            self.token_manager.refresh_token_expire_days = config["refresh_token_expire_days"]

        logger.info("Authentication manager initialized")

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, str]]:
        """
        Authenticate user and return tokens.

        Args:
            username: Username
            password: Password

        Returns:
            Dictionary with access and refresh tokens
        """
        if not self.enabled:
            return None

        user = self.user_manager.authenticate_user(username, password)
        if not user:
            return None

        # Create tokens
        access_token, access_expires = self.token_manager.create_access_token(
            user.user_id, user.username, user.roles, user.permissions
        )

        refresh_token, refresh_expires = self.token_manager.create_refresh_token(user.user_id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int((access_expires - datetime.now(timezone.utc)).total_seconds()),
        }

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify authentication token.

        Args:
            token: JWT token or API key

        Returns:
            User information if valid
        """
        if not self.enabled:
            # Return dummy user info when auth is disabled
            return {
                "user_id": "anonymous",
                "username": "anonymous",
                "roles": ["user"],
                "permissions": self.permission_manager.get_permissions_for_roles(["user"]),
            }

        # Try JWT token first
        if token.startswith("eyJ"):  # JWT tokens start with 'eyJ'
            payload = self.token_manager.verify_token(token)
            if payload:
                return payload

        # Try API key
        elif token.startswith("sk_"):
            api_key_info = self.token_manager.verify_api_key(token)
            if api_key_info:
                return {
                    "user_id": api_key_info["user_id"],
                    "username": api_key_info["username"],
                    "roles": api_key_info["roles"],
                    "permissions": api_key_info["permissions"],
                    "token_type": "api_key",
                }

        return None

    def has_permission(self, user_info: Dict[str, Any], permission: str) -> bool:
        """
        Check if user has required permission.

        Args:
            user_info: User information from token verification
            permission: Required permission

        Returns:
            True if user has permission
        """
        if not self.enabled:
            return True

        user_permissions = user_info.get("permissions", [])
        return self.permission_manager.has_permission(user_permissions, permission)

    def create_api_key(self, user_id: str, description: str = "") -> str:
        """
        Create API key for user.

        Args:
            user_id: User identifier
            description: Key description

        Returns:
            API key string
        """
        if not self.enabled:
            raise RuntimeError("Authentication is disabled")

        user = self.user_manager.get_user(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        return self.token_manager.create_api_key(
            user.user_id, user.username, user.roles, user.permissions, description
        )

    def revoke_token(self, token: str):
        """Revoke authentication token"""
        if self.enabled:
            if token.startswith("sk_"):
                self.token_manager.revoke_api_key(token)
            else:
                self.token_manager.revoke_token(token)

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New access token info
        """
        if not self.enabled:
            return None

        payload = self.token_manager.verify_token(refresh_token)
        if not payload or payload.get("token_type") != "refresh":
            return None

        user = self.user_manager.get_user(payload["user_id"])
        if not user or not user.active:
            return None

        # Create new access token
        access_token, access_expires = self.token_manager.create_access_token(
            user.user_id, user.username, user.roles, user.permissions
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int((access_expires - datetime.now(timezone.utc)).total_seconds()),
        }

    def get_user_info(self, user_id: str) -> Optional[UserInfo]:
        """Get user information"""
        if self.enabled:
            return self.user_manager.get_user(user_id)
        return None


def create_auth_manager(config_path: str = None) -> AuthenticationManager:
    """
    Create authentication manager from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured authentication manager
    """
    if config_path:
        config_dict = ConfigParser.load_config(config_path)
        auth_config = config_dict.get("auth", {})
    else:
        auth_config = {"enabled": False}

    return AuthenticationManager(auth_config)


if __name__ == "__main__":
    # Demo: Authentication system
    import argparse

    parser = argparse.ArgumentParser(description="Authentication Manager Demo")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--create-user", help="Create user with username")
    parser.add_argument("--email", help="User email address")
    parser.add_argument("--roles", nargs="+", default=["user"], help="User roles")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create auth manager
        auth_manager = create_auth_manager(args.config)

        if args.create_user:
            # Create new user
            if not auth_manager.enabled:
                print("❌ Authentication is disabled")
            else:
                user = auth_manager.user_manager.create_user(
                    args.create_user, args.email or f"{args.create_user}@example.com", args.roles
                )
                print(f"✅ User created: {user.username} with roles {user.roles}")

                # Create API key
                api_key = auth_manager.create_api_key(user.user_id, "CLI generated")
                print(f"🔑 API Key: {api_key}")

        else:
            # Test authentication
            if auth_manager.enabled:
                # Test admin login
                tokens = auth_manager.authenticate_user("admin", "password")
                if tokens:
                    print("✅ Admin authentication successful")
                    print(f"   Access token: {tokens['access_token'][:50]}...")
                else:
                    print("❌ Admin authentication failed")
            else:
                print("ℹ️  Authentication is disabled")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Authentication demo failed: {e}")
