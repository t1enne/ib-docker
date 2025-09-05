#!/usr/bin/env python3
"""
Interactive Brokers Client Portal API - Authentication Status Example
This script demonstrates how to check the authentication status of the gateway.
"""

import requests
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class IBKRGateway:
    def __init__(self, base_url="https://localhost:5000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/v1/api"

    def check_auth_status(self):
        """Check the authentication status of the gateway."""
        endpoint = "iserver/auth/status"
        url = f"{self.api_base}/{endpoint}"

        try:
            response = requests.get(url, verify=False, timeout=10)
            response.raise_for_status()

            status_data = response.json()

            print("Authentication Status:")
            print("-" * 30)
            print(f"Authenticated: {status_data.get('authenticated', False)}")
            print(f"Connected: {status_data.get('connected', False)}")
            print(f"Competing: {status_data.get('competing', False)}")
            print(f"Message: {status_data.get('message', 'N/A')}")

            if status_data.get("serverInfo"):
                print(f"Server Info: {status_data['serverInfo']}")

            return status_data

        except requests.exceptions.RequestException as e:
            print(f"Error checking authentication status: {e}")
            return None

    def tickle(self):
        """Send a tickle request to keep the session alive."""
        endpoint = "tickle"
        url = f"{self.api_base}/{endpoint}"

        try:
            response = requests.post(url, verify=False, timeout=10)
            response.raise_for_status()

            print("Session tickle successful")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error sending tickle: {e}")
            return False


def main():
    gateway = IBKRGateway()

    print("IBKR Client Portal Gateway - Authentication Status Check")
    print("=" * 55)

    # Check authentication status
    status = gateway.check_auth_status()

    if status:
        if status.get("authenticated") and status.get("connected"):
            print("\n✅ Gateway is authenticated and connected!")
            # Send a tickle to keep session alive
            # print("\nSending session tickle...")
            # gateway.tickle()

        else:
            print("\n❌ Gateway is not properly authenticated.")
            print("Please visit https://localhost:5000 to login.")
    else:
        print("\n❌ Unable to connect to gateway.")
        print("Make sure the gateway is running on https://localhost:5000")


if __name__ == "__main__":
    main()
