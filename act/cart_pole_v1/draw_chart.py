import matplotlib.pyplot as plt

from act.cart_pole_v1.constant import DATABASE_NAME, TABLE_NAME
from src.repositories.sqlite_utils import create_connection, get_all

if __name__ == '__main__':
    db_conn = create_connection(DATABASE_NAME)
    rows = get_all(db_conn, TABLE_NAME)
    db_conn.close()
    data = [row.score for row in rows]
    plt.plot(range(1, len(data) + 1), data)
    plt.show()
