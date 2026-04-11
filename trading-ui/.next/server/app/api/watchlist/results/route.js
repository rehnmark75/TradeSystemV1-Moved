"use strict";(()=>{var t={};t.id=131,t.ids=[131],t.modules={399:t=>{t.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:t=>{t.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},6371:(t,e,_)=>{_.r(e),_.d(e,{originalPathname:()=>w,patchFetch:()=>u,requestAsyncStorage:()=>m,routeModule:()=>p,serverHooks:()=>b,staticGenerationAsyncStorage:()=>E});var s={};_.r(s),_.d(s,{GET:()=>n,dynamic:()=>c});var r=_(7599),a=_(4294),o=_(4588),i=_(2921),l=_(740);let c="force-dynamic",d=["ema_50_crossover","ema_20_crossover","macd_bullish_cross"];async function n(t){let{searchParams:e}=new URL(t.url),_=e.get("watchlist"),s=e.get("date"),r=Number(e.get("limit")||100);if(!_)return i.NextResponse.json({error:"watchlist is required"},{status:400});let a=await l.d.connect();try{if(d.includes(_)){let t=`
        SELECT
          w.ticker,
          w.price,
          w.volume,
          w.avg_volume,
          w.ema_20,
          w.ema_50,
          w.ema_200,
          w.rsi_14,
          w.macd,
          w.gap_pct,
          w.price_change_1d,
          w.scan_date,
          w.crossover_date,
          (CURRENT_DATE - w.crossover_date) + 1 as days_on_list,
          w.avg_daily_change_5d,
          w.daq_score,
          w.daq_grade,
          w.daq_earnings_risk,
          w.daq_high_short_interest,
          w.rs_percentile,
          w.rs_trend,
          w.trade_ready,
          w.trade_ready_score,
          w.structure_rr_ratio,
          m.tv_overall_score,
          m.tv_overall_signal,
          m.perf_1w,
          m.perf_1m,
          m.perf_3m,
          i.analyst_rating,
          i.target_price,
          i.number_of_analysts,
          ar.period as reco_period,
          ar.strong_buy as reco_strong_buy,
          ar.buy as reco_buy,
          ar.hold as reco_hold,
          ar.sell as reco_sell,
          ar.strong_sell as reco_strong_sell,
          COALESCE(i.exchange, 'NASDAQ') as exchange,
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
          bt_closed.last_closed_side,
          w.bt_ema50_90d_signals,
          w.bt_ema50_90d_win_rate,
          w.bt_ema50_90d_avg_pnl,
          w.bt_ema50_90d_total_pnl,
          w.bt_ema50_90d_profit_factor,
          w.bt_ema50_90d_avg_hold_days,
          w.bt_ema50_90d_score,
          w.bt_ema50_90d_grade,
          w.bt_ema50_90d_confidence,
          w.bt_ema50_90d_supports_signal
        FROM stock_watchlist_results w
        LEFT JOIN stock_instruments i ON w.ticker = i.ticker
        LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
          AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        LEFT JOIN LATERAL (
          SELECT period, strong_buy, buy, hold, sell, strong_sell
          FROM stock_analyst_recommendations
          WHERE ticker = w.ticker
          ORDER BY period DESC
          LIMIT 1
        ) ar ON TRUE
        LEFT JOIN LATERAL (
          SELECT
            COUNT(*)::int AS trade_count,
            COUNT(*) FILTER (WHERE status = 'open')::int AS open_trade_count,
            MAX(open_time) FILTER (WHERE status = 'open') AS latest_open_time
          FROM broker_trades bt
          WHERE bt.ticker = w.ticker
             OR split_part(bt.ticker, '.', 1) = w.ticker
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
          WHERE bt.ticker = w.ticker
             OR split_part(bt.ticker, '.', 1) = w.ticker
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
          WHERE (bt.ticker = w.ticker OR split_part(bt.ticker, '.', 1) = w.ticker)
            AND bt.status = 'closed'
          ORDER BY bt.close_time DESC NULLS LAST
          LIMIT 1
        ) bt_closed ON TRUE
        WHERE w.watchlist_name = $1
          AND w.status = 'active'
        ORDER BY w.crossover_date DESC NULLS LAST, w.volume DESC
        LIMIT $2
      `,e=await a.query(t,[_,r]);return i.NextResponse.json({rows:e.rows})}let t=`
      SELECT
        w.ticker,
        w.price,
        w.volume,
        w.avg_volume,
        w.ema_20,
        w.ema_50,
        w.ema_200,
        w.rsi_14,
        w.macd,
        w.gap_pct,
        w.price_change_1d,
        w.scan_date,
        w.crossover_date,
        1 as days_on_list,
        w.avg_daily_change_5d,
        w.daq_score,
        w.daq_grade,
        w.daq_earnings_risk,
        w.daq_high_short_interest,
        w.rs_percentile,
        w.rs_trend,
        m.tv_overall_score,
        m.tv_overall_signal,
        m.perf_1w,
        m.perf_1m,
        m.perf_3m,
        i.analyst_rating,
        i.target_price,
        i.number_of_analysts,
        ar.period as reco_period,
        ar.strong_buy as reco_strong_buy,
        ar.buy as reco_buy,
        ar.hold as reco_hold,
        ar.sell as reco_sell,
        ar.strong_sell as reco_strong_sell,
        COALESCE(i.exchange, 'NASDAQ') as exchange,
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
        bt_closed.last_closed_side,
        w.bt_ema50_90d_signals,
        w.bt_ema50_90d_win_rate,
        w.bt_ema50_90d_avg_pnl,
        w.bt_ema50_90d_total_pnl,
        w.bt_ema50_90d_profit_factor,
        w.bt_ema50_90d_avg_hold_days
      FROM stock_watchlist_results w
      LEFT JOIN stock_instruments i ON w.ticker = i.ticker
      LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
        AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
      LEFT JOIN LATERAL (
        SELECT period, strong_buy, buy, hold, sell, strong_sell
        FROM stock_analyst_recommendations
        WHERE ticker = w.ticker
        ORDER BY period DESC
        LIMIT 1
      ) ar ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          COUNT(*)::int AS trade_count,
          COUNT(*) FILTER (WHERE status = 'open')::int AS open_trade_count,
          MAX(open_time) FILTER (WHERE status = 'open') AS latest_open_time
        FROM broker_trades bt
        WHERE bt.ticker = w.ticker
           OR split_part(bt.ticker, '.', 1) = w.ticker
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
        WHERE bt.ticker = w.ticker
           OR split_part(bt.ticker, '.', 1) = w.ticker
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
        WHERE (bt.ticker = w.ticker OR split_part(bt.ticker, '.', 1) = w.ticker)
          AND bt.status = 'closed'
        ORDER BY bt.close_time DESC NULLS LAST
        LIMIT 1
      ) bt_closed ON TRUE
      WHERE w.watchlist_name = $1
        AND w.scan_date = COALESCE($2::date, (
          SELECT MAX(scan_date)
          FROM stock_watchlist_results
          WHERE watchlist_name = $1
        ))
      ORDER BY w.scan_date DESC, w.volume DESC
      LIMIT $3
    `,e=await a.query(t,[_,s,r]);return i.NextResponse.json({rows:e.rows})}catch(t){return i.NextResponse.json({error:"Failed to load results"},{status:500})}finally{a.release()}}let p=new r.AppRouteRouteModule({definition:{kind:a.x.APP_ROUTE,page:"/api/watchlist/results/route",pathname:"/api/watchlist/results",filename:"route",bundlePath:"app/api/watchlist/results/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/watchlist/results/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:m,staticGenerationAsyncStorage:E,serverHooks:b}=p,w="/api/watchlist/results/route";function u(){return(0,o.patchFetch)({serverHooks:b,staticGenerationAsyncStorage:E})}},740:(t,e,_)=>{_.d(e,{d:()=>r}),function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}();let s=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",r=Object(function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}())({connectionString:s,max:10})}};var e=require("../../../../webpack-runtime.js");e.C(t);var _=t=>e(e.s=t),s=e.X(0,[5822,9967],()=>_(6371));module.exports=s})();