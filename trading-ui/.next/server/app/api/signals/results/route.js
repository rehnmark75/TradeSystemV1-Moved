"use strict";(()=>{var e={};e.id=6647,e.ids=[6647],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},1431:(e,t,s)=>{s.r(t),s.d(t,{originalPathname:()=>E,patchFetch:()=>h,requestAsyncStorage:()=>u,routeModule:()=>d,serverHooks:()=>g,staticGenerationAsyncStorage:()=>p});var a={};s.r(a),s.d(a,{GET:()=>m,dynamic:()=>l});var r=s(7599),_=s(4294),i=s(4588),n=s(2921),o=s(740);let c=["D","C","B","A","A+"],l="force-dynamic";async function m(e){let{searchParams:t}=new URL(e.url),s=t.get("scanner"),a=t.get("status"),r=t.get("minScore"),_=t.get("minClaudeGrade"),i="true"===t.get("claudeOnly"),l=t.get("claudeAction"),m=t.get("dateFrom"),d=t.get("dateTo"),u=t.get("minRs"),p=t.get("maxRs"),g=t.get("rsTrend"),E=Number(t.get("limit")||100),h=t.get("orderBy")||"date_desc",b=[],A=[];if(s&&(A.push(s),b.push(`scanner_name = $${A.length}`)),a&&(A.push(a),b.push(`status = $${A.length}`)),r&&(A.push(Number(r)),b.push(`composite_score >= $${A.length}`)),i&&b.push("claude_analyzed_at IS NOT NULL"),l&&(A.push(l),b.push(`claude_action = $${A.length}`)),_){let e=c.indexOf(_),t=c.filter((t,s)=>s>=e);b.push(`claude_grade = ANY($${A.length+1})`),A.push(t)}m&&(A.push(m),b.push(`DATE(signal_timestamp) >= $${A.length}`)),d&&(A.push(d),b.push(`DATE(signal_timestamp) <= $${A.length}`));let S=b.length?b.join(" AND "):"1=1",T=[];u&&T.push(`m.rs_percentile >= ${Number(u)}`),p&&T.push(`m.rs_percentile <= ${Number(p)}`),g&&T.push(`m.rs_trend = '${g}'`);let N=await o.d.connect();try{let e=`
      WITH latest_signals AS (
        SELECT DISTINCT ON (ticker, scanner_name)
          *
        FROM stock_scanner_signals
        WHERE ${S}
        ORDER BY ticker, scanner_name, signal_timestamp DESC
      )
      SELECT
        s.id,
        s.signal_timestamp,
        s.scanner_name,
        s.ticker,
        s.signal_type,
        s.entry_price,
        s.composite_score,
        s.quality_tier,
        s.status,
        s.trend_score,
        s.momentum_score,
        s.volume_score,
        s.pattern_score,
        s.risk_percent,
        s.risk_reward_ratio,
        s.setup_description,
        s.confluence_factors,
        s.timeframe,
        s.market_regime,
        s.claude_grade,
        s.claude_score,
        s.claude_action,
        s.claude_thesis,
        s.claude_key_strengths,
        s.claude_key_risks,
        s.claude_analyzed_at,
        s.news_sentiment_score,
        s.news_sentiment_level,
        s.news_headlines_count,
        i.name as company_name,
        i.sector,
        COALESCE(i.exchange, 'NASDAQ') as exchange,
        i.analyst_rating,
        i.target_price,
        i.number_of_analysts,
        -- RS and trade plan context
        m.rs_percentile,
        m.rs_trend,
        m.atr_14,
        m.atr_percent,
        m.swing_high,
        m.swing_low,
        m.swing_high_date,
        m.swing_low_date,
        m.relative_volume,
        -- TradingView summary counts
        m.tv_osc_buy,
        m.tv_osc_sell,
        m.tv_osc_neutral,
        m.tv_ma_buy,
        m.tv_ma_sell,
        m.tv_ma_neutral,
        m.tv_overall_signal,
        m.tv_overall_score,
        -- Oscillators and indicators
        m.rsi_14,
        m.stoch_k,
        m.stoch_d,
        m.cci_20,
        m.adx_14,
        m.plus_di,
        m.minus_di,
        m.ao_value,
        m.momentum_10,
        m.macd,
        m.macd_signal,
        m.stoch_rsi_k,
        m.stoch_rsi_d,
        m.williams_r,
        m.bull_power,
        m.bear_power,
        m.ultimate_osc,
        m.ema_10,
        m.ema_20,
        m.ema_30,
        m.ema_50,
        m.ema_100,
        m.ema_200,
        m.sma_10,
        m.sma_20,
        m.sma_30,
        m.sma_50,
        m.sma_100,
        m.sma_200,
        m.ichimoku_base,
        m.vwma_20,
        -- DAQ
        d.daq_score,
        d.daq_grade,
        d.mtf_score,
        d.volume_score as daq_volume_score,
        d.smc_score as daq_smc_score,
        d.quality_score as daq_quality_score,
        d.catalyst_score as daq_catalyst_score,
        d.news_score as daq_news_score,
        d.regime_score as daq_regime_score,
        d.sector_score as daq_sector_score,
        d.earnings_within_7d,
        d.high_short_interest,
        d.sector_underperforming,
        -- Earnings
        i.earnings_date,
        CASE
          WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
          THEN (i.earnings_date - CURRENT_DATE)
          ELSE NULL
        END as days_to_earnings,
        bt_summary.trade_count,
        bt_summary.open_trade_count,
        bt_summary.latest_open_time,
        bt_last.last_trade_status,
        bt_last.last_trade_open_time,
        bt_last.last_trade_close_time,
        bt_last.last_trade_profit,
        bt_last.last_trade_profit_pct,
        bt_last.last_trade_side,
        bt_closed.last_closed_time,
        bt_closed.last_closed_profit,
        bt_closed.last_closed_profit_pct,
        bt_closed.last_closed_side
      FROM latest_signals s
      LEFT JOIN stock_instruments i ON s.ticker = i.ticker
      LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
        AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
      LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
      LEFT JOIN LATERAL (
        SELECT
          COUNT(*)::int AS trade_count,
          COUNT(*) FILTER (WHERE status = 'open')::int AS open_trade_count,
          MAX(open_time) FILTER (WHERE status = 'open') AS latest_open_time
        FROM broker_trades bt
        WHERE bt.ticker = s.ticker
           OR split_part(bt.ticker, '.', 1) = s.ticker
      ) bt_summary ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          bt.status AS last_trade_status,
          bt.open_time AS last_trade_open_time,
          bt.close_time AS last_trade_close_time,
          bt.profit AS last_trade_profit,
          bt.profit_pct AS last_trade_profit_pct,
          bt.side AS last_trade_side
        FROM broker_trades bt
        WHERE bt.ticker = s.ticker
           OR split_part(bt.ticker, '.', 1) = s.ticker
        ORDER BY bt.open_time DESC NULLS LAST
        LIMIT 1
      ) bt_last ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          bt.close_time AS last_closed_time,
          bt.profit AS last_closed_profit,
          bt.profit_pct AS last_closed_profit_pct,
          bt.side AS last_closed_side
        FROM broker_trades bt
        WHERE (bt.ticker = s.ticker OR split_part(bt.ticker, '.', 1) = s.ticker)
          AND bt.status = 'closed'
        ORDER BY bt.close_time DESC NULLS LAST
        LIMIT 1
      ) bt_closed ON TRUE
    `;T.length&&(e+=` WHERE ${T.join(" AND ")}`),"timestamp"===h||"date_desc"===h?e+=" ORDER BY s.signal_timestamp DESC":"date_asc"===h?e+=" ORDER BY s.signal_timestamp ASC":e+=`
        ORDER BY
          CASE WHEN s.claude_analyzed_at IS NOT NULL THEN 0 ELSE 1 END,
          COALESCE(s.claude_score, 0) DESC,
          s.composite_score DESC,
          s.signal_timestamp DESC
      `,e+=` LIMIT ${E}`;let t=await N.query(e,A);return n.NextResponse.json({rows:t.rows})}catch(e){return n.NextResponse.json({error:"Failed to load signals"},{status:500})}finally{N.release()}}let d=new r.AppRouteRouteModule({definition:{kind:_.x.APP_ROUTE,page:"/api/signals/results/route",pathname:"/api/signals/results",filename:"route",bundlePath:"app/api/signals/results/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/signals/results/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:u,staticGenerationAsyncStorage:p,serverHooks:g}=d,E="/api/signals/results/route";function h(){return(0,i.patchFetch)({serverHooks:g,staticGenerationAsyncStorage:p})}},740:(e,t,s)=>{s.d(t,{d:()=>r}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",r=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),a=t.X(0,[5822,9967],()=>s(1431));module.exports=a})();