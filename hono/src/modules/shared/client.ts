import axios from "axios";
import https from "https";

export const client = axios.create({
  baseURL: "https://gateway:5000/v1/api/",
  httpsAgent: new https.Agent({ rejectUnauthorized: false }),
  timeout: 10000,
});
