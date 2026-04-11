"use strict";(()=>{var e={};e.id=4909,e.ids=[4909],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},5755:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>g,patchFetch:()=>y,requestAsyncStorage:()=>E,routeModule:()=>u,serverHooks:()=>m,staticGenerationAsyncStorage:()=>p});var s={};r.r(s),r.d(s,{GET:()=>d,dynamic:()=>_});var a=r(7599),o=r(4294),i=r(4588),n=r(2921),c=r(740);let _="force-dynamic",l={Technology:["Technology"],"Health Care":["Health Care","Healthcare"],Financials:["Financials","Financial Services"],"Consumer Discretionary":["Consumer Discretionary","Consumer Cyclical"],"Consumer Staples":["Consumer Staples","Consumer Defensive"],"Communication Services":["Communication Services"],Industrials:["Industrials"],Energy:["Energy"],Utilities:["Utilities"],"Real Estate":["Real Estate"],Materials:["Materials","Basic Materials"]};async function d(e){let{searchParams:t}=new URL(e.url),r=(t.get("ticker")||"").trim().toUpperCase();if(!r)return n.NextResponse.json({error:"ticker is required"},{status:400});let s=await c.d.connect();try{let e=await s.query(`
      SELECT
        ticker,
        name,
        exchange,
        sector,
        industry,
        market_cap,
        avg_volume,
        currency,
        earnings_date,
        dividend_yield,
        trailing_pe,
        forward_pe,
        profit_margin,
        revenue_growth,
        earnings_growth,
        debt_to_equity,
        current_ratio,
        quick_ratio,
        analyst_rating,
        target_price,
        target_high,
        target_low,
        number_of_analysts,
        fifty_two_week_high,
        fifty_two_week_low,
        fifty_two_week_change,
        fifty_day_average,
        two_hundred_day_average
      FROM stock_instruments
      WHERE ticker = $1
      LIMIT 1
      `,[r]),t=await s.query(`
      SELECT *
      FROM stock_screening_metrics
      WHERE ticker = $1
      ORDER BY calculation_date DESC
      LIMIT 1
      `,[r]),a=await s.query(`
      SELECT *
      FROM stock_watchlist_results
      WHERE ticker = $1
      ORDER BY COALESCE(scan_date, crossover_date) DESC NULLS LAST, crossover_date DESC NULLS LAST
      LIMIT 1
      `,[r]),o=await s.query(`
      SELECT
        s.*,
        d.daq_score,
        d.daq_grade,
        d.mtf_score,
        d.volume_score as daq_volume_score,
        d.smc_score as daq_smc_score,
        d.quality_score as daq_quality_score,
        d.catalyst_score as daq_catalyst_score,
        d.news_score as daq_news_score,
        d.regime_score as daq_regime_score,
        d.sector_score as daq_sector_score
      FROM stock_scanner_signals s
      LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
      WHERE s.ticker = $1
      ORDER BY s.signal_timestamp DESC
      LIMIT 1
      `,[r]),i=await s.query(`
      SELECT
        id,
        signal_timestamp,
        scanner_name,
        signal_type,
        composite_score,
        quality_tier,
        status,
        claude_action,
        claude_grade,
        news_sentiment_level,
        entry_price,
        risk_reward_ratio
      FROM stock_scanner_signals
      WHERE ticker = $1
      ORDER BY signal_timestamp DESC
      LIMIT 20
      `,[r]),c=await s.query(`
      SELECT period, strong_buy, buy, hold, sell, strong_sell
      FROM stock_analyst_recommendations
      WHERE ticker = $1
      ORDER BY period DESC
      LIMIT 1
      `,[r]),_=await s.query(`
      SELECT headline, summary, source, url, published_at, sentiment_score
      FROM stock_news_cache
      WHERE ticker = $1
      ORDER BY published_at DESC
      LIMIT 8
      `,[r]),d=e.rows[0]?.sector||null,u=null;if(d&&!(u=(await s.query(`
        SELECT
          sector,
          sector_return_1d,
          sector_return_5d,
          sector_return_20d,
          rs_vs_spy,
          rs_percentile,
          rs_trend,
          stocks_in_sector,
          pct_above_sma50,
          pct_bullish_trend,
          sector_stage
        FROM sector_analysis
        WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
          AND sector = $1
        LIMIT 1
        `,[d])).rows[0]||null)){let e=l[d]||[];e.length&&(u=(await s.query(`
            SELECT
              sector,
              sector_return_1d,
              sector_return_5d,
              sector_return_20d,
              rs_vs_spy,
              rs_percentile,
              rs_trend,
              stocks_in_sector,
              pct_above_sma50,
              pct_bullish_trend,
              sector_stage
            FROM sector_analysis
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
              AND sector = ANY($1::text[])
            LIMIT 1
            `,[e])).rows[0]||null)}let E=await s.query(`
      SELECT *
      FROM v_current_market_regime
      LIMIT 1
      `);return n.NextResponse.json({instrument:e.rows[0]||null,metrics:t.rows[0]||null,watchlist:a.rows[0]||null,signal:o.rows[0]||null,signal_history:i.rows||[],analyst:c.rows[0]||null,news:_.rows||[],sector_context:u,market_regime:E.rows[0]||null})}catch(e){return console.error("stock detail error",e),n.NextResponse.json({error:"Failed to load stock detail",detail:String(e)},{status:500})}finally{s.release()}}let u=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/stocks/detail/route",pathname:"/api/stocks/detail",filename:"route",bundlePath:"app/api/stocks/detail/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/stocks/detail/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:E,staticGenerationAsyncStorage:p,serverHooks:m}=u,g="/api/stocks/detail/route";function y(){return(0,i.patchFetch)({serverHooks:m,staticGenerationAsyncStorage:p})}},740:(e,t,r)=>{r.d(t,{d:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let s=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:s,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),s=t.X(0,[5822,9967],()=>r(5755));module.exports=s})();