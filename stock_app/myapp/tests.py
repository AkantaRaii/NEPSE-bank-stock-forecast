from django.test import TestCase, Client
from django.urls import reverse
import numpy as np

class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        """Test the home view to ensure it returns a 200 response and renders the correct template."""
        response = self.client.get(reverse('home'))  # Replace 'home' with the actual URL name
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')

    def test_stock_view(self):
        """
        Test the stock view to ensure it returns a 200 response,
        renders the correct template, and calls the `predict` function.
        """
        stock_name = 'ADBL'
        response = self.client.get(reverse('stock', args=[stock_name]))  # Replace 'stock' with the actual URL name
        
        # Assert that the view returns a 200 response
        self.assertEqual(response.status_code, 200)
        
        # Assert that the correct template is rendered
        self.assertTemplateUsed(response, 'stock.html')

        # Assert that the context contains the correct values
        self.assertEqual(response.context['stock_name'], stock_name)

        # Assert that predicted_value is of type numpy.ndarray
        self.assertIsInstance(response.context['predicted_value'], np.ndarray)
