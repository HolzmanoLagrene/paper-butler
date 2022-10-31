from django.test import TestCase

import datetime
from basic.models import Invoice, Receipt


# Create your tests here.

class InvoiceTest(TestCase):

    def setUp(self):
        self.inv_1 = Invoice.objects.create(name_of_invoicer="test", price=300, deadline_date=datetime.date.today())
        self.inv_2 = Invoice.objects.create(name_of_invoicer="test", price=300, deadline_date=datetime.date.today())

    def test_equality(self):
        """Two objects are never the same"""
        self.assertNotEqual(self.inv_1, self.inv_2)

    def test_default_payment(self):
        """Payment Flag is always negative"""
        self.assertFalse(self.inv_1.payed)

class ReceiptTest(TestCase):

    def setUp(self):
        self.rec_1 = Receipt.objects.create(buying_date=datetime.date.today(), price=300, name_of_article="TestArticel", name_of_store="TestStore",)
        self.rec_2 = Receipt.objects.create(buying_date=datetime.date.today(), price=300, name_of_article="TestArticel", name_of_store="TestStore",)

    def test_equality(self):
        """Two objects are never the same"""
        self.assertNotEqual(self.rec_1, self.rec_2)

    def test_date_of_expiry_accessor(self):
        """Payment Flag is always negative"""
        self.assertEqual(self.rec_1.date_of_expiry,self.rec_1.buying_date+datetime.timedelta(days=self.rec_1.days_of_warranty))