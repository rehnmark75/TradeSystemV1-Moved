"use strict";(()=>{var e={};e.id=8533,e.ids=[8533],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4038:(e,a,s)=>{s.r(a),s.d(a,{originalPathname:()=>N,patchFetch:()=>u,requestAsyncStorage:()=>g,routeModule:()=>l,serverHooks:()=>d,staticGenerationAsyncStorage:()=>m});var n={};s.r(n),s.d(n,{GET:()=>p,dynamic:()=>o});var r=s(7599),t=s(4294),i=s(4588),_=s(2921),c=s(1577);let o="force-dynamic";function E(e){if(!e)return null;let a=new Date(e);return Number.isNaN(a.valueOf())?null:a}async function p(e){let{searchParams:a}=new URL(e.url),s=E(a.get("start")),n=E(a.get("end")),r=function(e){if(!e)return 1;let a=Number(e);return!Number.isFinite(a)||a<=0?1:a}(a.get("days")),t=n??new Date,i=s??new Date(new Date(t).setDate(t.getDate()-r));try{let e=(await c.B.query(`
      SELECT
        COUNT(*) as total_scans,
        COUNT(DISTINCT scan_cycle_id) as scan_cycles,
        COUNT(DISTINCT epic) as unique_epics,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals_generated,
        SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
        AVG(raw_confidence) as avg_raw_confidence,
        AVG(final_confidence) as avg_final_confidence,
        AVG(CASE WHEN signal_generated THEN final_confidence END) as avg_signal_confidence,
        COUNT(DISTINCT rejection_reason) as rejection_types,
        SUM(CASE WHEN rejection_reason = 'confidence' THEN 1 ELSE 0 END) as confidence_rejections,
        SUM(CASE WHEN rejection_reason = 'dedup' THEN 1 ELSE 0 END) as dedup_rejections,
        MIN(scan_timestamp) as first_scan,
        MAX(scan_timestamp) as last_scan
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      `,[i,t])).rows[0]??{},a=Number(e.total_scans??0),s=Number(e.signals_generated??0),n={total_scans:a,scan_cycles:Number(e.scan_cycles??0),unique_epics:Number(e.unique_epics??0),signals_generated:s,buy_signals:Number(e.buy_signals??0),sell_signals:Number(e.sell_signals??0),avg_raw_confidence:Number(e.avg_raw_confidence??0),avg_final_confidence:Number(e.avg_final_confidence??0),avg_signal_confidence:Number(e.avg_signal_confidence??0),rejection_types:Number(e.rejection_types??0),confidence_rejections:Number(e.confidence_rejections??0),dedup_rejections:Number(e.dedup_rejections??0),first_scan:e.first_scan,last_scan:e.last_scan,signal_rate:a?s/a:0},r=await c.B.query(`
      SELECT
        DATE_TRUNC('hour', scan_timestamp) as hour,
        COUNT(*) as total_scans,
        COUNT(DISTINCT scan_cycle_id) as scan_cycles,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        AVG(raw_confidence) as avg_confidence,
        AVG(atr_pips) as avg_atr
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      GROUP BY DATE_TRUNC('hour', scan_timestamp)
      ORDER BY hour
      `,[i,t]),o=await c.B.query(`
      SELECT
        market_regime,
        COUNT(*) as count,
        AVG(regime_confidence) as avg_confidence,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        AVG(CASE WHEN signal_generated THEN final_confidence END) as avg_signal_confidence
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
        AND market_regime IS NOT NULL
      GROUP BY market_regime
      ORDER BY count DESC
      `,[i,t]),E=await c.B.query(`
      SELECT
        session,
        session_volatility,
        COUNT(*) as count,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        AVG(raw_confidence) as avg_confidence,
        AVG(atr_pips) as avg_atr_pips,
        AVG(spread_pips) as avg_spread
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
        AND session IS NOT NULL
      GROUP BY session, session_volatility
      ORDER BY count DESC
      `,[i,t]),p=await c.B.query(`
      SELECT
        rejection_reason,
        COUNT(*) as count,
        AVG(raw_confidence) as avg_raw_confidence,
        AVG(final_confidence) as avg_final_confidence,
        AVG(confidence_threshold) as avg_threshold,
        COUNT(DISTINCT epic) as affected_epics
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
        AND rejection_reason IS NOT NULL
      GROUP BY rejection_reason
      ORDER BY count DESC
      `,[i,t]),l=await c.B.query(`
      SELECT
        epic,
        pair_name,
        COUNT(*) as total_scans,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
        AVG(raw_confidence) as avg_raw_confidence,
        AVG(final_confidence) as avg_final_confidence,
        AVG(rsi_14) as avg_rsi,
        AVG(adx) as avg_adx,
        AVG(atr_pips) as avg_atr_pips,
        AVG(spread_pips) as avg_spread,
        MODE() WITHIN GROUP (ORDER BY market_regime) as dominant_regime,
        MODE() WITHIN GROUP (ORDER BY volatility_state) as dominant_volatility,
        SUM(CASE WHEN rejection_reason = 'confidence' THEN 1 ELSE 0 END) as confidence_rejections,
        SUM(CASE WHEN rejection_reason = 'dedup' THEN 1 ELSE 0 END) as dedup_rejections
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      GROUP BY epic, pair_name
      ORDER BY signals DESC, total_scans DESC
      `,[i,t]),g=await c.B.query(`
      SELECT
        signal_generated,
        AVG(rsi_14) as avg_rsi,
        AVG(adx) as avg_adx,
        AVG(efficiency_ratio) as avg_er,
        AVG(atr_pips) as avg_atr,
        AVG(bb_width_percentile) as avg_bb_percentile,
        AVG(smart_money_score) as avg_smc_score,
        AVG(mtf_confluence_score) as avg_mtf_score,
        AVG(entry_quality_score) as avg_entry_quality,
        COUNT(*) as count
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      GROUP BY signal_generated
      `,[i,t]),m={signals:null,non_signals:null};for(let e of g.rows??[])m[e.signal_generated?"signals":"non_signals"]={avg_rsi:Number(e.avg_rsi??0),avg_adx:Number(e.avg_adx??0),avg_er:Number(e.avg_er??0),avg_atr:Number(e.avg_atr??0),avg_bb_percentile:Number(e.avg_bb_percentile??0),avg_smc_score:Number(e.avg_smc_score??0),avg_mtf_score:Number(e.avg_mtf_score??0),avg_entry_quality:Number(e.avg_entry_quality??0),count:Number(e.count??0)};return _.NextResponse.json({range:{start:i,end:t},summary:n,timeline:r.rows??[],regimes:o.rows??[],sessions:E.rows??[],rejections:p.rows??[],epics:l.rows??[],indicators:m})}catch(e){return console.error("Failed to load performance snapshot",e),_.NextResponse.json({error:"Failed to load performance snapshot"},{status:500})}}let l=new r.AppRouteRouteModule({definition:{kind:t.x.APP_ROUTE,page:"/api/forex/performance-snapshot/route",pathname:"/api/forex/performance-snapshot",filename:"route",bundlePath:"app/api/forex/performance-snapshot/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/performance-snapshot/route.ts",nextConfigOutput:"standalone",userland:n}),{requestAsyncStorage:g,staticGenerationAsyncStorage:m,serverHooks:d}=l,N="/api/forex/performance-snapshot/route";function u(){return(0,i.patchFetch)({serverHooks:d,staticGenerationAsyncStorage:m})}},1577:(e,a,s)=>{s.d(a,{B:()=>r}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let n=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",r=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:n,max:10})}};var a=require("../../../../webpack-runtime.js");a.C(e);var s=e=>a(a.s=e),n=a.X(0,[5822,9967],()=>s(4038));module.exports=n})();