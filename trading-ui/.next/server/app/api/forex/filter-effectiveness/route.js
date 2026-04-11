"use strict";(()=>{var t={};t.id=859,t.ids=[859],t.modules={399:t=>{t.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:t=>{t.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4782:(t,e,a)=>{a.r(e),a.d(e,{originalPathname:()=>u,patchFetch:()=>p,requestAsyncStorage:()=>m,routeModule:()=>_,serverHooks:()=>f,staticGenerationAsyncStorage:()=>c});var r={};a.r(r),a.d(r,{GET:()=>N,dynamic:()=>l});var s=a(7599),i=a(4294),E=a(4588),o=a(2921),n=a(1577);let l="force-dynamic";async function N(t){let{searchParams:e}=new URL(t.url),a=function(t){if(!t)return 30;let e=Number(t);return!Number.isFinite(e)||e<=0?30:e}(e.get("days")),r=new Date;r.setDate(r.getDate()-a);try{let t=await n.B.query(`SELECT
        'market_regime' as filter_name,
        COALESCE(a.performance_metrics->>'market_regime', 'Unknown') as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY COALESCE(a.performance_metrics->>'market_regime', 'Unknown')
      ORDER BY trades DESC`,[r]),e=await n.B.query(`SELECT
        'volatility_state' as filter_name,
        COALESCE(a.performance_metrics->>'volatility_state', 'Unknown') as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY COALESCE(a.performance_metrics->>'volatility_state', 'Unknown')
      ORDER BY trades DESC`,[r]),s=await n.B.query(`SELECT
        'structure_bias' as filter_name,
        COALESCE(a.market_structure_analysis->>'current_bias', 'Unknown') as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY COALESCE(a.market_structure_analysis->>'current_bias', 'Unknown')
      ORDER BY trades DESC`,[r]),i=await n.B.query(`SELECT
        'direction_alignment' as filter_name,
        CASE
          WHEN a.market_structure_analysis->>'current_bias' = 'RANGING' THEN 'RANGING'
          WHEN (t.direction = 'BUY' AND a.market_structure_analysis->>'current_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.market_structure_analysis->>'current_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.market_structure_analysis->>'current_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'COUNTER'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN a.market_structure_analysis->>'current_bias' = 'RANGING' THEN 'RANGING'
          WHEN (t.direction = 'BUY' AND a.market_structure_analysis->>'current_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.market_structure_analysis->>'current_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.market_structure_analysis->>'current_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'COUNTER'
        END
      ORDER BY win_rate DESC NULLS LAST`,[r]),E=await n.B.query(`SELECT
        'order_flow_alignment' as filter_name,
        CASE
          WHEN (t.direction = 'BUY' AND a.order_flow_analysis->>'order_flow_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.order_flow_analysis->>'order_flow_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.order_flow_analysis->>'order_flow_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'CONFLICTING'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN (t.direction = 'BUY' AND a.order_flow_analysis->>'order_flow_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.order_flow_analysis->>'order_flow_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.order_flow_analysis->>'order_flow_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'CONFLICTING'
        END
      ORDER BY trades DESC`,[r]),l=await n.B.query(`SELECT
        'entry_quality' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.3 THEN 'Low (0.3-0.5)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float < 0.3 THEN 'Very Low (<0.3)'
          ELSE 'Unknown'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.3 THEN 'Low (0.3-0.5)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float < 0.3 THEN 'Very Low (<0.3)'
          ELSE 'Unknown'
        END
      ORDER BY
        CASE CASE
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.3 THEN 'Low (0.3-0.5)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float < 0.3 THEN 'Very Low (<0.3)'
          ELSE 'Unknown'
        END
          WHEN 'High (>=0.7)' THEN 1
          WHEN 'Medium (0.5-0.7)' THEN 2
          WHEN 'Low (0.3-0.5)' THEN 3
          WHEN 'Very Low (<0.3)' THEN 4
          ELSE 5
        END`,[r]),N=await n.B.query(`SELECT
        'efficiency_ratio' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.5 THEN 'High (>=0.5)'
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.3 THEN 'Medium (0.3-0.5)'
          ELSE 'Low (<0.3)'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
        AND a.performance_metrics->>'efficiency_ratio' IS NOT NULL
      GROUP BY CASE
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.5 THEN 'High (>=0.5)'
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.3 THEN 'Medium (0.3-0.5)'
          ELSE 'Low (<0.3)'
        END
      ORDER BY trades DESC`,[r]),_=await n.B.query(`SELECT
        'mtf_alignment' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'all_timeframes_aligned')::boolean = true THEN 'All TFs Aligned'
          ELSE 'Not Aligned'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN (a.performance_metrics->>'all_timeframes_aligned')::boolean = true THEN 'All TFs Aligned'
          ELSE 'Not Aligned'
        END
      ORDER BY trades DESC`,[r]),m=await n.B.query(`SELECT
        'mtf_confluence' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          ELSE 'Low (<0.5)'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
        AND a.performance_metrics->>'mtf_confluence_score' IS NOT NULL
      GROUP BY CASE
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          ELSE 'Low (<0.5)'
        END
      ORDER BY trades DESC`,[r]),c=(await n.B.query(`SELECT
        COUNT(*)::int as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log
      WHERE timestamp >= $1 AND LOWER(status) = 'closed'`,[r])).rows[0]||{total_trades:0,wins:0,losses:0,win_rate:0,total_pnl:0,avg_pnl:0},f=(t,e,a)=>{let r=a.filter(t=>t.trades>=5);if(r.length<2)return{name:t,description:e,metrics:a,recommendation:"Insufficient data for analysis",is_predictive:!1};let s=r.map(t=>t.win_rate||0),i=Math.max(...s),E=Math.min(...s),o=r.find(t=>t.win_rate===i),n=r.find(t=>t.win_rate===E),l=i-E>=10,N="";return l?o&&n&&(N=`Consider blocking "${n.filter_value}" (${n.win_rate}% WR) and favoring "${o.filter_value}" (${o.win_rate}% WR)`):N="Not predictive - no significant performance difference between groups",{name:t,description:e,metrics:a,recommendation:N,is_predictive:l}},u=[f("Entry Quality Score","Signal entry quality based on Fib zone and candle momentum",l.rows),f("Direction vs Structure Alignment","Whether trade direction matches market structure bias",i.rows),f("Market Structure Bias","Current market structure from Smart Money analysis",s.rows),f("Order Flow Alignment","Trade direction vs order flow bias",E.rows),f("Market Regime","Detected market regime at signal time",t.rows),f("Volatility State","Market volatility state at signal time",e.rows),f("MTF Alignment","Multi-timeframe directional alignment",_.rows),f("Efficiency Ratio","Price movement efficiency (trend strength)",N.rows),f("MTF Confluence Score","Overall multi-timeframe confluence score",m.rows)];return o.NextResponse.json({baseline:c,filterGroups:u,days:a,generatedAt:new Date().toISOString()})}catch(t){return console.error("Failed to load filter effectiveness",t),o.NextResponse.json({error:"Failed to load filter effectiveness analysis"},{status:500})}}let _=new s.AppRouteRouteModule({definition:{kind:i.x.APP_ROUTE,page:"/api/forex/filter-effectiveness/route",pathname:"/api/forex/filter-effectiveness",filename:"route",bundlePath:"app/api/forex/filter-effectiveness/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/filter-effectiveness/route.ts",nextConfigOutput:"standalone",userland:r}),{requestAsyncStorage:m,staticGenerationAsyncStorage:c,serverHooks:f}=_,u="/api/forex/filter-effectiveness/route";function p(){return(0,E.patchFetch)({serverHooks:f,staticGenerationAsyncStorage:c})}},1577:(t,e,a)=>{a.d(e,{B:()=>s}),function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}();let r=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",s=Object(function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}())({connectionString:r,max:10})}};var e=require("../../../../webpack-runtime.js");e.C(t);var a=t=>e(e.s=t),r=e.X(0,[5822,9967],()=>a(4782));module.exports=r})();