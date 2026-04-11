"use strict";(()=>{var e={};e.id=7919,e.ids=[7919],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},3895:(e,s,a)=>{a.r(s),a.d(s,{originalPathname:()=>E,patchFetch:()=>c,requestAsyncStorage:()=>N,routeModule:()=>u,serverHooks:()=>v,staticGenerationAsyncStorage:()=>d});var t={};a.r(t),a.d(t,{GET:()=>n,dynamic:()=>m});var r=a(7599),p=a(4294),i=a(4588),_=a(2921),l=a(1577);let m="force-dynamic";function o(e){return e?e.replace("CS.D.","").replace(".MINI.IP","").replace(".CEEM.IP",""):""}async function n(e){let{searchParams:s}=new URL(e.url),a=function(e){if(!e)return 7;let s=Number(e);return!Number.isFinite(s)||s<=0?7:s}(s.get("days")),t=new Date;t.setDate(t.getDate()-a);try{let e=((await l.B.query(`
      SELECT
        id,
        symbol,
        direction,
        entry_price,
        timestamp,
        status,
        profit_loss,
        vsl_peak_profit_pips as mfe_pips,
        vsl_mae_pips as mae_pips,
        vsl_mae_price as mae_price,
        vsl_mae_timestamp as mae_time,
        virtual_sl_pips,
        vsl_stage,
        vsl_breakeven_triggered as hit_breakeven,
        vsl_stage1_triggered as hit_stage1,
        vsl_stage2_triggered as hit_stage2
      FROM trade_log
      WHERE is_scalp_trade = true
      AND timestamp >= $1
      ORDER BY timestamp DESC
      `,[t])).rows??[]).map(e=>{var s;let a=null==e.profit_loss?null:Number(e.profit_loss),t=null==e.mae_pips?null:Number(e.mae_pips),r=null==e.virtual_sl_pips?null:Number(e.virtual_sl_pips),p=null!=t&&null!=r&&0!==r?t/r*100:null;return{...e,entry_price:null==e.entry_price?null:Number(e.entry_price),profit_loss:a,mfe_pips:null==e.mfe_pips?null:Number(e.mfe_pips),mae_pips:t,virtual_sl_pips:r,mae_pct_of_vsl:null==p?null:Number(p.toFixed(1)),symbol_short:o(e.symbol),result:(s=e.status,null!=a?a>0?"WIN":a<0?"LOSS":"BREAKEVEN":"tracking"===s?"OPEN":"PENDING")}}),s=((await l.B.query(`
      SELECT
        symbol,
        COUNT(*) as total_trades,
        AVG(vsl_mae_pips) as avg_mae_pips,
        MAX(vsl_mae_pips) as max_mae_pips,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vsl_mae_pips) as median_mae_pips,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY vsl_mae_pips) as p75_mae_pips,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY vsl_mae_pips) as p90_mae_pips,
        AVG(vsl_peak_profit_pips) as avg_mfe_pips,
        MAX(vsl_peak_profit_pips) as max_mfe_pips,
        AVG(virtual_sl_pips) as avg_vsl_setting,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses
      FROM trade_log
      WHERE is_scalp_trade = true
      AND timestamp >= $1
      AND vsl_mae_pips IS NOT NULL
      GROUP BY symbol
      ORDER BY total_trades DESC
      `,[t])).rows??[]).map(e=>{let s=Number(e.total_trades??0),a=Number(e.wins??0);return{...e,symbol_short:o(e.symbol),total_trades:s,wins:a,losses:Number(e.losses??0),win_rate:s?a/s*100:0,avg_mae_pips:Number(e.avg_mae_pips??0),max_mae_pips:Number(e.max_mae_pips??0),median_mae_pips:Number(e.median_mae_pips??0),p75_mae_pips:Number(e.p75_mae_pips??0),p90_mae_pips:Number(e.p90_mae_pips??0),avg_mfe_pips:Number(e.avg_mfe_pips??0),max_mfe_pips:Number(e.max_mfe_pips??0),avg_vsl_setting:Number(e.avg_vsl_setting??0)}});return _.NextResponse.json({trades:e,summary:s})}catch(e){return console.error("Failed to load MAE analysis",e),_.NextResponse.json({error:"Failed to load MAE analysis"},{status:500})}}let u=new r.AppRouteRouteModule({definition:{kind:p.x.APP_ROUTE,page:"/api/forex/mae-analysis/route",pathname:"/api/forex/mae-analysis",filename:"route",bundlePath:"app/api/forex/mae-analysis/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/mae-analysis/route.ts",nextConfigOutput:"standalone",userland:t}),{requestAsyncStorage:N,staticGenerationAsyncStorage:d,serverHooks:v}=u,E="/api/forex/mae-analysis/route";function c(){return(0,i.patchFetch)({serverHooks:v,staticGenerationAsyncStorage:d})}},1577:(e,s,a)=>{a.d(s,{B:()=>r}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let t=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",r=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:t,max:10})}};var s=require("../../../../webpack-runtime.js");s.C(e);var a=e=>s(s.s=e),t=s.X(0,[5822,9967],()=>a(3895));module.exports=t})();