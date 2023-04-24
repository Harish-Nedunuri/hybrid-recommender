"""
Contains class JDBCDatabase and JDBCConfig.
It is responsible for establishing connection with JDBC databses, like MS SQL Server.
"""
import pyspark.sql
from survival_classifier_training.analytics_core.database_mgmt.spark_handler import (
    SparkMixin,
)
from survival_classifier_training.analytics_core.env_config.environment_security import (
    DatabricksSecretEnvironmentConfig,
)
from pydantic import Field


class JDBCConfig(DatabricksSecretEnvironmentConfig):
    _secret_scope: str = Field(env_var="SECRET_SCOPE")
    jdbc_string: str = Field(env_var="SQL_JDBC_URL")
    password: str = Field(env_var_secret_key="SQL_PASSWORD_KEY")


class JDBCDatabase(SparkMixin):
    """
    Object responsible for establishing connection with JDBC databses, like MS SQL Server.
    """

    def __init__(self, config: DatabricksSecretEnvironmentConfig):
        super().__init__()
        self.config_ = config
        self.views_ = list()

    def read(self, table: str) -> pyspark.sql.DataFrame:
        """
        Perform an SQL query to the database.

        :param table: Table name
        """
        return (
            self.spark.read.format("jdbc")
            .option("url", self.config_.jdbc_string)
            .option("dbtable", table)
            .option("password", self.config_.password)
            .load()
        )

    def create_view(self, table: str, view_name: str, local: bool = True):
        df = self.read(table)
        if local:
            df.createOrReplaceTempView(view_name)
        else:
            df.createOrReplaceGlobalTempView(view_name)

    def remove_views(self):
        for view in self.views_:
            self.spark.catalog.dropTempView(view)
        self.views_ = list()

    def write(self, table: str, df: pyspark.sql.DataFrame):
        properties = {"password": self.config_.password}
        return df.write.jdbc(
            url=self.config_.jdbc_string,
            table=table,
            mode="append",
            properties=properties,
        )

# jdbc_config = JDBCConfig.read_env_vars(dbutils, secret_scope=os.environ['SECRET_SCOPE'])
# sql_db = JDBCDatabase(jdbc_config)
# sql_db.create_view('dbo.Routes', 'Routes')