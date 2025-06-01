# PDF Q&A Bot

A web application that allows users to upload PDF documents and ask questions about their content using AI.

## Features

- PDF document upload
- Interactive web interface
- AI-powered question answering
- Real-time responses

## Prerequisites

- Node.js (v14 or higher)
- OpenAI API key

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the server:
   ```bash
   npm start
   ```
2. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Click the "Choose File" button to select a PDF document
2. Click "Upload" to process the document
3. Once the upload is complete, you can start asking questions about the document's content
4. Type your question in the text area and click "Ask"
5. The AI will analyze the document and provide an answer based on the content

## Note

Make sure you have a valid OpenAI API key and sufficient credits in your account. The application uses GPT-4 for generating responses. 