import Fastify from "fastify";
import { OpenRouterService } from "./openrouterService.ts";

export const createServer = (routerService: OpenRouterService ) => {
    const app = Fastify({ logger: false })

    app.post('/chat', {
        schema: {
            body: {
                type: 'object',
                required: ['question'],
                properties: {
                    question: { type: 'string', minLength: 5 }
                }
            }
        }
    }, async (request, reply) => {
        try {
            const { question } = request.body as { question: string }
            const response = await routerService.generate(question)
            reply.send(response)
        } catch (error) {
            console.log('Error handling /chat request: ', error)
            reply.code(500)
        }
    })

    return app
}