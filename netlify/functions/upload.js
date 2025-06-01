import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAIEmbeddings } from "@langchain/openai";
import { setVectorStore } from './shared.js';

// Initialize embeddings
const embeddings = new OpenAIEmbeddings();

export const handler = async (event, context) => {
    // Handle CORS preflight requests
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            }
        };
    }

    // Only allow POST
    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({ error: 'Method not allowed' })
        };
    }

    try {
        // Check if the request is multipart/form-data
        if (!event.headers['content-type']?.includes('multipart/form-data')) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                body: JSON.stringify({ error: 'Content type must be multipart/form-data' })
            };
        }

        // Parse the multipart form data
        const boundary = event.headers['content-type'].split('boundary=')[1];
        const parts = event.body.split('--' + boundary);
        
        // Find the file part
        const filePart = parts.find(part => part.includes('filename='));
        if (!filePart) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                body: JSON.stringify({ error: 'No file uploaded' })
            };
        }

        // Extract the file content
        const fileContent = filePart.split('\r\n\r\n')[1].split('\r\n')[0];
        if (!fileContent) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                body: JSON.stringify({ error: 'Invalid file content' })
            };
        }

        const fileBuffer = Buffer.from(fileContent, 'base64');

        // Load and process the PDF
        const loader = new PDFLoader(fileBuffer);
        const docs = await loader.load();

        // Split the documents into chunks
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splitDocs = await textSplitter.splitDocuments(docs);

        // Create the vector store and save it to shared state
        const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
        setVectorStore(vectorStore);

        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({ message: 'PDF processed successfully' })
        };
    } catch (error) {
        console.error('Error processing PDF:', error);
        return {
            statusCode: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({ error: 'Error processing PDF: ' + error.message })
        };
    }
}; 