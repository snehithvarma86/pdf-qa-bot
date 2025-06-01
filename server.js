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
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : '*'
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Configure multer for memory storage instead of disk storage
const upload = multer({ 
    storage: multer.memoryStorage(),
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/pdf') {
            cb(null, true);
        } else {
            cb(new Error('Only PDF files are allowed'));
        }
    },
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB limit
    }
});

// Serve static files
app.use(express.static(__dirname));
app.use(express.json());

// Initialize embeddings
const embeddings = new OpenAIEmbeddings();

// Store for the current vector store
let currentVectorStore = null;

// Error handling middleware
const errorHandler = (err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        error: process.env.NODE_ENV === 'production' 
            ? 'Internal server error' 
            : err.message 
    });
};

app.post('/upload', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // Load and process the PDF from buffer
        const loader = new PDFLoader(req.file.buffer);
        const docs = await loader.load();

        // Split the documents into chunks
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const splitDocs = await textSplitter.splitDocuments(docs);

        // Create the vector store
        currentVectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

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

// Apply error handling middleware
app.use(errorHandler);

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
}); 