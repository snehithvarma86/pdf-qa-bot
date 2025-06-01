import { ChatOpenAI } from "@langchain/openai";
import { getVectorStore } from './shared.js';

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
        const { question } = JSON.parse(event.body);
        const currentVectorStore = getVectorStore();
        
        if (!currentVectorStore) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                body: JSON.stringify({ error: 'Please upload a PDF first' })
            };
        }

        if (!question) {
            return {
                statusCode: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                body: JSON.stringify({ error: 'No question provided' })
            };
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

        return {
            statusCode: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({ answer: response.content })
        };
    } catch (error) {
        console.error('Error processing query:', error);
        return {
            statusCode: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            body: JSON.stringify({ error: 'Error processing query' })
        };
    }
}; 