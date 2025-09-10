import axios from "axios";
import https from "https";

export const client = axios.create({
  baseURL: "https://localhost:5000/v1/api/",
  timeout: 10000,
  httpsAgent: new https.Agent({ rejectUnauthorized: false }),
});
