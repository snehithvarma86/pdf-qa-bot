import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAIEmbeddings } from "@langchain/openai";
import multer from 'multer';
import { IncomingForm } from 'formidable';
import { promises as fs } from 'fs';

// Initialize embeddings
const embeddings = new OpenAIEmbeddings();

// Store for the current vector store (Note: this will reset on cold starts)
let currentVectorStore = null;

export const handler = async (event, context) => {
    // Only allow POST
    if (event.httpMethod !== 'POST') {
        return {
            statusCode: 405,
            body: JSON.stringify({ error: 'Method not allowed' })
        };
    }

    try {
        // Parse the multipart form data
        const form = new IncomingForm();
        const { fields, files } = await new Promise((resolve, reject) => {
            form.parse(event.body, (err, fields, files) => {
                if (err) reject(err);
                resolve({ fields, files });
            });
        });

        const file = files.pdf;
        if (!file) {
            return {
                statusCode: 400,
                body: JSON.stringify({ error: 'No file uploaded' })
            };
        }

        // Read the file
        const fileBuffer = await fs.readFile(file.path);

        // Load and process the PDF
        const loader = new PDFLoader(fileBuffer);
        const docs = await loader.load();

        // Split the documents into chunks
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splitDocs = await textSplitter.splitDocuments(docs);

        // Create the vector store
        currentVectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

        // Clean up the temporary file
        await fs.unlink(file.path);

        return {
            statusCode: 200,
            body: JSON.stringify({ message: 'PDF processed successfully' })
        };
    } catch (error) {
        console.error('Error processing PDF:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Error processing PDF' })
        };
    }
}; 