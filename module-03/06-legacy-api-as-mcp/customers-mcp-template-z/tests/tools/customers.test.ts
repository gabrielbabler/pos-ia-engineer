import { describe, it, after, before } from 'node:test'
import assert from 'node:assert'
import { createTestClient } from '../helpers.ts'
import { Client } from '@modelcontextprotocol/sdk/client'
import { type CustomerMutation, type Customer } from '../../src/domain/customer.ts'

type CustomerResult = { structuredContent: { customers: Customer[] } }
type CustomerMutationResult = { structuredContent: CustomerMutation }

describe('Customer MCP Suite', () => {
    let client: Client
    before(async () => {
        client = await createTestClient()
    })

    after(async () => {
        await client.close()
    })

    it('should list all customers', async () => {
        const result = await client.callTool({
            name: 'list_customers',
            arguments: {}
        }) as unknown as CustomerResult

        assert.ok(
            Array.isArray(result.structuredContent.customers),
            'Should return an array of customers'
        )
    })

    it('should create a customer', async () => {
        const customer = {
            name: "Jhon",
            phone: "123456789"
        }
        const result = await client.callTool({
            name: 'create_customer',
            arguments: customer
        }) as unknown as CustomerMutationResult

        assert.ok(result.structuredContent.id,
            'Should contain id'
        )
        assert.deepStrictEqual(
            result.structuredContent.message,
            `user ${customer.name} created!`,
        )
    })

    it('should create a customer', async () => {
        const customer = {
            name: "Jhon",
            phone: "123456789"
        }
        
        const { structuredContent: { id } } = await client.callTool({
            name: 'create_customer',
            arguments: customer
        }) as unknown as CustomerMutationResult
        
        const result = await client.callTool({
            name: 'get_customer',
            arguments: {
                name: "Jhon"
            }
        }) as unknown as CustomerMutationResult

        assert.ok(result.structuredContent.customer?._id,
            'Should contain id'
        )
        assert.deepStrictEqual(
            result.structuredContent.customer.name,
            customer.name,
        )
    })
})