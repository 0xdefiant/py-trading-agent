# This is a summary of all the code for this tutorial
import os
from coinbase.rest import RESTClient
from json import dumps

api_key = os.environ["COINBASE_API_KEY"]
api_secret = os.environ["COINBASE_API_SECRET"]

client = RESTClient(api_key=api_key, api_secret=api_secret)

accounts = client.get_accounts()
# print("Accounts:", dumps(accounts.to_dict(), indent=2))

listPortfolio = client.get_portfolios()
# print("List Portfolio:", dumps(listPortfolio.to_dict(), indent=2))
uuid = listPortfolio.to_dict()["portfolios"][0]["uuid"]
# print("UUID:", uuid)

portfolioBreakdown = client.get_portfolio_breakdown(portfolio_uuid=uuid)
def print_portfolio_summary(portfolio_breakdown):
    breakdown = portfolio_breakdown["breakdown"]
    portfolio = breakdown["portfolio"]
    balances = breakdown["portfolio_balances"]
    positions = breakdown["spot_positions"]

    print(f"Portfolio: {portfolio['name']} (UUID: {portfolio['uuid']})")
    print(f"Total Balance: {balances['total_balance']['value']} {balances['total_balance']['currency']}")
    print("\nAssets:")
    for pos in positions:
        print(f"  - {pos['asset']}:")
        print(f"      Total: {pos['total_balance_crypto']} {pos['asset']} (${pos['total_balance_fiat']:.2f})")
        print(f"      Allocation: {pos['allocation']*100:.2f}%")
        print(f"      Unrealized PnL: {pos['unrealized_pnl']:.2f} USD")
        print(f"      Account Type: {pos['account_type']}")

print_portfolio_summary(portfolioBreakdown.to_dict())