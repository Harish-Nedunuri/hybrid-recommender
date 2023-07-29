from pydantic import BaseModel
import os
from hybrid_recommender.AnalyticsCore.audit.logging import (logger)


class EnvironmentConfig(BaseModel):
    """
    Pydantic class which implements the extra functionality of read its values from environment variables.
    """

    @classmethod
    def read_env_vars(cls):
        """
        Reads environment variables and creates a class object based on it.
        """
        # Iterate over fields to add
        fields = cls.__dict__["__fields__"]
        configs = dict()
        for f in fields.values():
            extra = f.field_info.extra
            if "env_var" in extra:
                env_var_name = extra["env_var"]
            else:
                env_var_name = f.name
            configs[f.name] = os.getenv(env_var_name)
        return cls(**configs)


class DatabricksSecretEnvironmentConfig(BaseModel):
    """
    Pydantic class which implements the extra functionality of read its values from environment variables,
    and also read secrets based on provided secret scope.
    """

    @classmethod
    def read_env_vars(cls, dbutils, secret_scope: str):
        """
        Reads environment variables and creates a class object based on it.
        """
        # Iterate over fields to add
        fields = cls.__dict__["__fields__"]

        configs = dict()
        for f in fields.values():
            if f.name == "_secret_scope":
                continue
            extra = f.field_info.extra
            if "env_var" in extra:
                configs[f.name] = os.getenv(extra["env_var"])
            elif "env_var_secret_key" in extra:
                secret_key = os.getenv(extra["env_var_secret_key"])
                secret_value = dbutils.secrets.get(scope=secret_scope, key=secret_key)
                if secret_value is None:
                    logger.warning(
                        "Failed to read secret key {} from scope {}".format(
                            secret_key, secret_scope
                        )
                    )
                    continue
                configs[f.name] = secret_value
        return cls(**configs)
