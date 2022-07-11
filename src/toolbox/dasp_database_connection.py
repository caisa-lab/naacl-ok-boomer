import logging
import json
from abc import ABC

try:
    import psycopg2
except ModuleNotFoundError as e:
    print("Psycopg2 not installed")

try:
    from sshtunnel import SSHTunnelForwarder
except ModuleNotFoundError as e:
    print("Ssh tunnel not installed")


config_path = "config.json"

try:
    with open(config_path) as f:
        config = json.load(f)

    use_ssh = config['use_ssh']

    ssh_credentials = {'server': config['ssh_server'], 'port': int(config['ssh_port']), 'username': config['ssh_username'],
                   'password': config['ssh_password']}

    db_credentials = {'database': config['pg_db'], 'username': config['pg_user'], 'password': config['pg_password']}
except Exception as e:
    print("Found no/flawed database config. This is only a problem if you want to use the dasp database!")

class DaspDatabaseConnection(object):
    """
    Abstraction for connecting to the ukp database.
    """
    def __init__(self):
        """
        Constructor
        """
        self.__server = None
        self.__conn = None
        self.__curs = None
        self.open_connection()

    def open_connection(self):
        """
        Open the connection to the database.
        This function only works if the ssh tunnel was priorly opened
        with e.g. the terminal.
        """
        try:
            if use_ssh:
                self.__server = SSHTunnelForwarder(
                    (ssh_credentials['server'], ssh_credentials['port']),
                    ssh_username=ssh_credentials['username'],
                    ssh_password=ssh_credentials['password'],
                    remote_bind_address=('localhost', 5432))
                self.__server.start()

            params = {
                'database': db_credentials['database'],
                'user': db_credentials['username'],
                'password': db_credentials['password'],
                'host': 'localhost',
                'port': 5432
            }

            self.__conn = psycopg2.connect(**params)
            logging.debug("Connected to Server!")
            self.__curs = self.__conn.cursor()
            logging.info("Connected to Server!")
            logging.debug("Connected to database {}".format(db_credentials['database']))
        except Exception as e:
            logging.error("Failed to connect to database {} \n{}".format(db_credentials['database'], e))
            raise e

    def close_connection(self):
        """
        Close the connection to the database.
        """
        if self.__conn is not None:
            self.__conn.close()

        if self.__server is not None:
            self.__server.stop()

    def __execute(self, sql_statement, values):
        """
        Execute a given sql statement with a given set of values.
        Only works if the connection was priorly opened.
        :param sql_statement: The sql statement to execute
        :param values: Additional values
        :return: The rows that were fetched from the database as a list of tuples
        """
        rows = []
        if self.__conn is not None and self.__curs is not None:
            try:
                self.__curs.execute(sql_statement, values)
                rows = self.__curs.fetchall()
                self.__conn.commit()
            except (Exception, psycopg2.DatabaseError) as e:
                logging.error(e)
                self.__conn.rollback()
        return rows

    def get(self, sql_statement):
        """
        Function for exectuing a sql statement without values.
        :param sql_statement: The sql statement to execute
        :return: The rows that were fetched from the database as a list of tuples
        """
        return self.__execute(sql_statement, [])

class DatabaseAccessor(object):

    def __init__(self, origin):
        self.origin = origin
        self.db = None

    def init_db_connection(self):
        self.db = DaspDatabaseConnection()

    def close_db_connection(self):
        self.db.close_connection()
        self.db = None

    def safe_db_queue(self, db_q):
        #logging.info('Trying to sql queue: ' + db_q)
        print('Trying to sql queue: ' + db_q)
        try:
            res = self.db.get(db_q)
        except Exception as e:
            logging.error(e)
            self.db.close_connection()
            return []
        return res
