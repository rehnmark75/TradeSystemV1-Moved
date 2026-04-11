"use strict";(()=>{var e={};e.id=117,e.ids=[117],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},6676:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>g,patchFetch:()=>f,requestAsyncStorage:()=>c,routeModule:()=>u,serverHooks:()=>N,staticGenerationAsyncStorage:()=>d});var a={};r.r(a),r.d(a,{GET:()=>m,dynamic:()=>o});var s=r(7599),i=r(4294),_=r(4588),n=r(2921),p=r(1577);let o="force-dynamic";function l(e){if(null==e||""===e)return null;let t=Number(e);return Number.isFinite(t)?t:null}async function m(e){let{searchParams:t}=new URL(e.url),r=function(e){if(!e)return 7;let t=Number(e);return!Number.isFinite(t)||t<=0?7:t}(t.get("days")),a=new Date;a.setDate(a.getDate()-r);try{let e=((await p.B.query(`
      SELECT
        t.id,
        t.symbol,
        t.direction,
        t.entry_price,
        t.timestamp as trade_timestamp,
        t.status,
        t.profit_loss,
        t.vsl_peak_profit_pips as mfe_pips,
        t.vsl_mae_pips as mae_pips,
        t.vsl_mae_timestamp as mae_timestamp,
        t.virtual_sl_pips,
        t.vsl_stage,
        t.closed_at,
        t.is_scalp_trade,
        a.id as alert_id,
        a.alert_timestamp as signal_timestamp,
        a.price as signal_price,
        a.confidence_score,
        a.signal_trigger,
        a.trigger_type,
        a.strategy_indicators->'tier3_entry'->>'entry_type' as entry_type,
        a.strategy_indicators->'tier3_entry'->>'order_type' as order_type,
        a.strategy_indicators->'tier3_entry'->>'limit_offset_pips' as limit_offset_pips,
        a.strategy_indicators->'tier3_entry'->>'pullback_depth' as pullback_depth,
        a.strategy_indicators->'tier3_entry'->>'in_optimal_zone' as in_optimal_zone,
        a.pattern_type,
        a.pattern_strength,
        a.rsi_divergence_detected,
        a.rsi_divergence,
        a.htf_candle_direction,
        a.market_session
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'tracking', 'expired')
      ORDER BY t.timestamp DESC
      `,[a])).rows??[]).map(e=>{let t=null==e.profit_loss?null:Number(e.profit_loss),r=null==e.entry_price?null:Number(e.entry_price),a=null==e.signal_price?null:Number(e.signal_price),s=null==e.mfe_pips?null:Number(e.mfe_pips),i=null==e.mae_pips?null:Number(e.mae_pips),_=e.symbol??"",n=e.direction??"",p=e.status??"",o=null;if(null!=r&&null!=a){let e=_.includes("JPY")?.01:1e-4,t=(r-a)/e;"SELL"===n&&(t=-t),o=Number.isFinite(t)?t:null}let m=null;if(e.mae_timestamp&&e.trade_timestamp){let t=new Date(e.mae_timestamp).getTime(),r=new Date(e.trade_timestamp).getTime();Number.isFinite(t)&&Number.isFinite(r)&&(m=(t-r)/1e3)}return{...e,entry_price:r,profit_loss:t,signal_price:a,mfe_pips:s,mae_pips:i,symbol_short:_?_.replace("CS.D.","").replace(".MINI.IP","").replace(".CEEM.IP",""):"",result:null!=t?t>0?"WIN":t<0?"LOSS":"BREAKEVEN":"tracking"===p?"OPEN":"PENDING",zero_mfe:(s??0)<.5,slippage_pips:o,time_to_mae_seconds:m,limit_offset_pips:l(e.limit_offset_pips),pullback_depth:l(e.pullback_depth),confidence_score:l(e.confidence_score)}}),t=((await p.B.query(`
      SELECT
        COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', 'UNKNOWN') as entry_type,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
        COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
        COALESCE(SUM(t.profit_loss), 0) as total_pnl,
        AVG(t.vsl_mae_pips) as avg_mae_pips,
        AVG(t.vsl_peak_profit_pips) as avg_mfe_pips,
        COUNT(CASE WHEN COALESCE(t.vsl_peak_profit_pips, 0) < 0.5 THEN 1 END) as zero_mfe_count,
        AVG(a.confidence_score) as avg_confidence,
        AVG(CAST(a.strategy_indicators->'tier3_entry'->>'pullback_depth' AS FLOAT)) as avg_pullback_depth
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'expired')
      AND t.profit_loss IS NOT NULL
      GROUP BY a.strategy_indicators->'tier3_entry'->>'entry_type'
      ORDER BY total_trades DESC
      `,[a])).rows??[]).map(e=>{let t=Number(e.total_trades??0),r=Number(e.wins??0),a=Number(e.zero_mfe_count??0);return{...e,total_trades:t,wins:r,losses:Number(e.losses??0),avg_pnl:Number(e.avg_pnl??0),total_pnl:Number(e.total_pnl??0),avg_mae_pips:Number(e.avg_mae_pips??0),avg_mfe_pips:Number(e.avg_mfe_pips??0),zero_mfe_count:a,avg_confidence:Number(e.avg_confidence??0),avg_pullback_depth:Number(e.avg_pullback_depth??0),win_rate:t?r/t*100:0,zero_mfe_pct:t?a/t*100:0}}),r=((await p.B.query(`
      SELECT
        COALESCE(NULLIF(a.signal_trigger, ''), 'STANDARD') as signal_trigger,
        COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', 'UNKNOWN') as entry_type,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
        COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
        COALESCE(SUM(t.profit_loss), 0) as total_pnl,
        AVG(t.vsl_mae_pips) as avg_mae_pips,
        AVG(t.vsl_peak_profit_pips) as avg_mfe_pips,
        COUNT(CASE WHEN COALESCE(t.vsl_peak_profit_pips, 0) < 0.5 THEN 1 END) as zero_mfe_count,
        AVG(a.confidence_score) as avg_confidence
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'expired')
      AND t.profit_loss IS NOT NULL
      GROUP BY a.signal_trigger, a.strategy_indicators->'tier3_entry'->>'entry_type'
      ORDER BY total_trades DESC
      `,[a])).rows??[]).map(e=>{let t=Number(e.total_trades??0),r=Number(e.wins??0),a=Number(e.zero_mfe_count??0);return{...e,total_trades:t,wins:r,losses:Number(e.losses??0),avg_pnl:Number(e.avg_pnl??0),total_pnl:Number(e.total_pnl??0),avg_mae_pips:Number(e.avg_mae_pips??0),avg_mfe_pips:Number(e.avg_mfe_pips??0),zero_mfe_count:a,avg_confidence:Number(e.avg_confidence??0),win_rate:t?r/t*100:0,zero_mfe_pct:t?a/t*100:0}});return n.NextResponse.json({trades:e,summary:t,by_trigger:r})}catch(e){return console.error("Failed to load entry timing analysis",e),n.NextResponse.json({error:"Failed to load entry timing analysis"},{status:500})}}let u=new s.AppRouteRouteModule({definition:{kind:i.x.APP_ROUTE,page:"/api/forex/entry-timing/route",pathname:"/api/forex/entry-timing",filename:"route",bundlePath:"app/api/forex/entry-timing/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/entry-timing/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:c,staticGenerationAsyncStorage:d,serverHooks:N}=u,g="/api/forex/entry-timing/route";function f(){return(0,_.patchFetch)({serverHooks:N,staticGenerationAsyncStorage:d})}},1577:(e,t,r)=>{r.d(t,{B:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),a=t.X(0,[5822,9967],()=>r(6676));module.exports=a})();