import z from "zod";

export const CustomerSchema = z.object({
    _id: z.string().optional(),
    name: z.string(),
    phone: z.string(),
})

export type Customer = z.infer<typeof CustomerSchema>

export const CustomerMutationSchema = z.object({
    id: z.string().optional().describe("MongoDB ObjectId of the customer"),
    message: z.string().optional().describe('Confirmation message'),
    isError: z.boolean().optional().describe('Indicates if an error occurred'),

    customer: CustomerSchema.optional().describe("The found customer"),
    customers: z.array(CustomerSchema).optional().describe("List of customers"),
})

export type CustomerMutation = z.infer<typeof CustomerMutationSchema>