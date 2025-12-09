#!/usr/bin/env python3
"""
Test script for StockChartGenerator
Run inside Docker: python3 /app/stock_scanner/test_chart_generation.py
"""
import sys
sys.path.insert(0, '/app')

import asyncio
from stock_scanner.services import StockChartGenerator
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner import config


async def test_chart_generation():
    """Test chart generation with a real signal"""

    # Connect to database
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    # Initialize chart generator with database
    gen = StockChartGenerator(db_manager=db)
    print(f"Chart generator available: {gen.is_available}")

    # Get a recent signal to test with
    signal_query = """
        SELECT ticker, entry_price, stop_loss, take_profit_1, take_profit_2,
               signal_type, scanner_name, quality_tier, composite_score
        FROM stock_scanner_signals
        WHERE quality_tier IN ('A+', 'A')
        ORDER BY signal_timestamp DESC
        LIMIT 1
    """
    signal_row = await db.fetchrow(signal_query)

    if signal_row:
        signal = dict(signal_row)
        ticker = signal['ticker']
        print(f"\nTesting chart generation for: {ticker}")
        print(f"Signal: {signal['signal_type']} at ${signal['entry_price']:.2f}")
        print(f"Stop: ${signal['stop_loss']:.2f}, Target: ${signal['take_profit_1']:.2f}")

        # Generate chart
        chart_base64 = await gen.generate_signal_chart(
            ticker=ticker,
            signal=signal
        )

        if chart_base64:
            print(f"\n✅ Chart generated successfully!")
            print(f"Base64 length: {len(chart_base64)} characters")

            # Save to file for visual inspection
            import base64
            img_data = base64.b64decode(chart_base64)
            output_path = f'/tmp/{ticker}_chart.png'
            with open(output_path, 'wb') as f:
                f.write(img_data)
            print(f"Chart saved to: {output_path}")
        else:
            print("❌ Chart generation failed!")
    else:
        print("No signals found to test with")

    await db.close()


if __name__ == '__main__':
    asyncio.run(test_chart_generation())
