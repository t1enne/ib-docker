# Interactive Brokers Client Portal Gateway Docker Container

This Docker container provides a convenient way to run the Interactive Brokers Client Portal Gateway for API access. The Client Portal Gateway enables programmatic access to IBKR's trading platform through RESTful APIs.

## Features

- 🐳 **Dockerized**: Easy deployment and consistent environment
- 🔒 **SSL Support**: Self-signed certificates for secure connections
- 🔧 **Configurable**: Environment variables for easy customization
- 🏥 **Health Checks**: Built-in container health monitoring
- 📊 **Examples**: Python examples for common API operations

## Prerequisites

- Docker and Docker Compose installed
- Interactive Brokers account (Live or Paper Trading)
- Basic understanding of IBKR Client Portal API

## Quick Start

1. **Clone or download this repository**
2. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```
3. **Access the gateway:**
   - Open your browser to https://localhost:5000
   - Accept the SSL certificate warning (expected for self-signed cert)
   - Login with your IBKR credentials

4. **Verify authentication:**
   ```bash
   curl -k https://localhost:5000/v1/api/iserver/auth/status
   ```

## Container Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key variables:
- `JAVA_OPTS`: JVM memory settings (default: `-Xmx512m`)
- `GATEWAY_PORT`: Gateway port (default: `5000`)
- `SESSION_TIMEOUT`: Session timeout in seconds (default: `86400`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### SSL Certificates

The container uses the default IBKR certificate. To generate a custom self-signed certificate:

```bash
./generate-cert.sh
```

This creates certificates in the `./certs` directory that can be mounted into the container.

## Usage Examples

### Python Authentication Check

```bash
cd examples/python
pip install -r requirements.txt
python auth_status.py
```

### Basic API Calls

After authentication, you can make API calls:

```bash
# Check authentication status
curl -k https://localhost:5000/v1/api/iserver/auth/status

# Get account information
curl -k https://localhost:5000/v1/api/portfolio/accounts

# Keep session alive
curl -k -X POST https://localhost:5000/v1/api/tickle
```

## Docker Commands

### Build the image:
```bash
docker build -t ib-gateway .
```

### Run with Docker Compose:
```bash
docker-compose up -d
```

### View logs:
```bash
docker-compose logs -f ib-gateway
```

### Stop the container:
```bash
docker-compose down
```

## File Structure

```
ib-docker/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker Compose configuration
├── conf.yaml              # Gateway configuration
├── entrypoint.sh           # Container startup script
├── generate-cert.sh        # SSL certificate generator
├── .env.example           # Environment variables template
├── examples/
│   └── python/
│       ├── auth_status.py  # Authentication example
│       └── requirements.txt
├── certs/                 # SSL certificates (generated)
└── data/                  # Persistent data (mounted)
```

## Important Notes

### Authentication Requirements

- **Manual Login Required**: You must login manually via the web interface at https://localhost:5000
- **Single Session**: Cannot be logged in elsewhere (TWS, Client Portal web, etc.)
- **2FA Support**: Multi-factor authentication is supported
- **Session Management**: Sessions last 24 hours with proper tickle requests

### Paper Trading

To use Paper Trading:
1. Set up Paper Trading credentials in Client Portal
2. Use Paper Trading username/password when logging in
3. API calls will operate on the paper account

### Security Considerations

- **SSL Warnings**: Self-signed certificates will show browser warnings (expected)
- **Localhost Only**: Default configuration binds to localhost for security
- **Network Access**: Ensure proper firewall rules if exposing beyond localhost

## Troubleshooting

### Common Issues

1. **401 Unauthorized**: Not properly authenticated - visit https://localhost:5000 to login
2. **Connection Refused**: Gateway not running - check container logs
3. **SSL Errors**: Expected with self-signed certificates - use `-k` flag with curl

### Debug Mode

Enable debug logging:
```bash
docker-compose exec ib-gateway tail -f /app/gateway/clientportal.gw/logs/ibgateway.log
```

### Health Check

Check container health:
```bash
docker-compose ps
```

## API Documentation

- **IBKR API Docs**: https://interactivebrokers.github.io/cpwebapi/
- **Quickstart Guide**: https://interactivebrokers.github.io/cpwebapi/quickstart
- **API Reference**: https://interactivebrokers.github.io/cpwebapi/webapi_doc

## Support

For IBKR API support:
- 📖 **Documentation**: https://interactivebrokers.com/campus/ibkr-api-page/
- 🎫 **Support Tickets**: Through IBKR Client Portal
- 🗣️ **Forums**: IBKR API Community forums

## License

This project is provided as-is for educational and development purposes. Interactive Brokers' terms of service apply to all API usage.

---

**⚠️ Disclaimer**: This is an unofficial Docker container. Always refer to official IBKR documentation for the most current API information and requirements.