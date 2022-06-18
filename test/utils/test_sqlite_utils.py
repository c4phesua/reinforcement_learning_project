import os
from unittest import TestCase

from src.repositories.sqlite_utils import create_connection, create_table, insert_data, get_all, update_data, \
    get_not_evaluate_oldest_weight, get_latest_weight


class TestSqliteUtils(TestCase):
    def setUp(self) -> None:
        print('setup')
        self.conn = create_connection('test_file.db')

    def tearDown(self) -> None:
        print('tear down')
        self.conn.close()
        path = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'training_results', 'test_file.db')
        os.remove(path)

    def test_create_connection(self):
        path = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'training_results', 'test_file.db')
        self.assertTrue(os.path.isfile(path))

    def test_create_table(self):
        create_table(self.conn, 'test_table')
        c = self.conn.cursor()
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='test_table' ''')
        self.assertEqual(c.fetchone()[0], 1)
        c.close()

    def test_insert_data(self):
        create_table(self.conn, 'test_table')
        insert_data("a/b/c", 12, self.conn, 'test_table')
        data = get_all(self.conn, 'test_table')
        self.assertEqual('a/b/c', data[0].weight_file)
        self.assertEqual(12, data[0].episode)

    def test_update_data(self):
        create_table(self.conn, 'test_table')
        insert_data("a/b/c", 12, self.conn, 'test_table')
        data = get_all(self.conn, 'test_table')
        self.assertEqual(None, data[0].score)
        update_data(1, 100, self.conn, 'test_table')
        data = get_all(self.conn, 'test_table')
        self.assertEqual(100, data[0].score)

    def test_get_not_evaluate_latest_weight(self):
        create_table(self.conn, 'test_table')
        insert_data("test0", 12, self.conn, 'test_table')
        insert_data("test1", 12, self.conn, 'test_table')
        insert_data("test2", 15, self.conn, 'test_table')
        update_data(1, 100, self.conn, 'test_table')
        data = get_not_evaluate_oldest_weight(self.conn, 'test_table')
        self.assertEqual("test1", data.weight_file)

    def test_get_latest_weight(self):
        create_table(self.conn, 'test_table')
        insert_data("test1", 12, self.conn, 'test_table')
        insert_data("test2", 15, self.conn, 'test_table')
        update_data(2, 100, self.conn, 'test_table')
        data = get_latest_weight(self.conn, 'test_table')
        self.assertEqual("test2", data.weight_file)

    def test_get_all(self):
        create_table(self.conn, 'test_table')
        data = get_all(self.conn, 'test_table')
        self.assertListEqual([], data)

    def test_get_all_with_data(self):
        create_table(self.conn, 'test_table')
        insert_data("test1", 12, self.conn, 'test_table')
        insert_data("test2", 15, self.conn, 'test_table')
        insert_data("test3", 15, self.conn, 'test_table')
        data = get_all(self.conn, 'test_table')
        self.assertEqual(1, data[0].id)
        self.assertEqual(2, data[1].id)
        self.assertEqual(3, data[2].id)
