import express from 'express';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;

// Configure multer for PDF upload
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname)
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/pdf') {
            cb(null, true);
        } else {
            cb(new Error('Only PDF files are allowed'));
        }
    }
});

// Serve static files
app.use(express.static(__dirname));
app.use(express.json());

// Initialize embeddings
const embeddings = new OpenAIEmbeddings();

// Store for the current vector store
let currentVectorStore = null;

app.post('/upload', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // Load and process the PDF
        const loader = new PDFLoader(req.file.path);
        const docs = await loader.load();

        // Split the documents into chunks
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splitDocs = await textSplitter.splitDocuments(docs);

        // Create and save the vector store
        currentVectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
        await currentVectorStore.save("faiss_store");

        res.json({ message: 'PDF processed successfully' });
    } catch (error) {
        console.error('Error processing PDF:', error);
        res.status(500).json({ error: 'Error processing PDF' });
    }
});

app.post('/query', async (req, res) => {
    try {
        if (!currentVectorStore) {
            return res.status(400).json({ error: 'Please upload a PDF first' });
        }

        const { question } = req.body;
        if (!question) {
            return res.status(400).json({ error: 'No question provided' });
        }

        const relevantChunks = await currentVectorStore.similaritySearch(question, 4);
        const context = relevantChunks
            .map((chunk) => chunk.pageContent)
            .join("\n---\n");

        const model = new ChatOpenAI({
            temperature: 0,
            modelName: "gpt-4",
        });

        const prompt = `Given the following context from a PDF document:
${context}

Answer the question: "${question}"
If the answer is not in the context, say "I don't know."`;

        const response = await model.invoke(prompt);

        res.json({ answer: response.content });
    } catch (error) {
        console.error('Error processing query:', error);
        res.status(500).json({ error: 'Error processing query' });
    }
});

// Create uploads directory if it doesn't exist
import { mkdir } from 'fs/promises';
try {
    await mkdir('uploads', { recursive: true });
} catch (error) {
    console.error('Error creating uploads directory:', error);
}

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
}); 