import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { registerListCustomersTool } from "./tools/listCustomers.ts";
import { CustomerService } from "../application/customerService.ts";
import { registerApiInfoResource } from "./resources/apiInfo.ts";
import { registerCreateCustomersTool } from "./tools/createCustomer.ts";

const BASE_URL = "http://localhost:9999/v1";
const service = new CustomerService(BASE_URL)

export const server = new McpServer({
    name: "@gbabler/gb-customers-mcp",
    version: "0.0.1",
});

registerListCustomersTool(server, service)
registerApiInfoResource(server, BASE_URL)
registerCreateCustomersTool(server, service)