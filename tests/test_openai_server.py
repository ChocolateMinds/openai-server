import unittest
from unittest.mock import patch, Mock
from openai_server import load_pdfs, split_text_into_chunks  # Import the functions you want to test

class TestApp(unittest.TestCase):

    @patch('app.PdfReader')
    def test_load_pdfs(self, MockPdfReader):
        # Mock PdfReader and its methods
        mock_pdf = Mock()
        mock_pdf.pages = [Mock(extract_text="Sample text")]
        MockPdfReader.return_value = mock_pdf

        # Call the function
        knowledge_base = load_pdfs("./data/pdfs")

        # Assertions to check if the function behaves as expected
        self.assertIsNotNone(knowledge_base)
        # Add more assertions based on your specific requirements

    def test_split_text_into_chunks(self):
        # Sample text
        text = "This is a sample text for testing."

        # Call the function
        chunks = split_text_into_chunks(text)

        # Assertions to check if the function behaves as expected
        self.assertEqual(len(chunks), 1)  # Assuming the text is small enough to fit into one chunk
        self.assertEqual(chunks[0], text)
        # Add more assertions based on your specific requirements

if __name__ == '__main__':
    unittest.main()
