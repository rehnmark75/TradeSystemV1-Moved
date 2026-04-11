"use strict";(()=>{var e={};e.id=8341,e.ids=[8341],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},7518:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>m,patchFetch:()=>C,requestAsyncStorage:()=>u,routeModule:()=>_,serverHooks:()=>N,staticGenerationAsyncStorage:()=>E});var s={};r.r(s),r.d(s,{GET:()=>d,dynamic:()=>p});var a=r(7599),o=r(4294),n=r(4588),l=r(2921),i=r(1577);let p="force-dynamic";async function d(e){let{searchParams:t}=new URL(e.url),r=function(e){if(!e)return 30;let t=Number(e);return!Number.isFinite(t)||t<=0?30:t}(t.get("days")),s=new Date;s.setDate(s.getDate()-r);try{let e=((await i.B.query(`
      SELECT
        a.strategy,
        COUNT(t.*) as total_trades,
        COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
        COALESCE(SUM(t.profit_loss), 0) as total_pnl,
        COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
        COALESCE(AVG(a.confidence_score), 0) as avg_confidence,
        COALESCE(MAX(t.profit_loss), 0) as best_trade,
        COALESCE(MIN(t.profit_loss), 0) as worst_trade,
        COUNT(DISTINCT t.symbol) as pairs_traded
      FROM trade_log t
      INNER JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      GROUP BY a.strategy
      ORDER BY total_pnl DESC
      `,[s])).rows??[]).map(e=>{let t=Number(e.total_trades??0),r=Number(e.wins??0);return{...e,total_trades:t,wins:r,losses:Number(e.losses??0),total_pnl:Number(e.total_pnl??0),avg_pnl:Number(e.avg_pnl??0),avg_confidence:Number(e.avg_confidence??0),best_trade:Number(e.best_trade??0),worst_trade:Number(e.worst_trade??0),pairs_traded:Number(e.pairs_traded??0),win_rate:t>0?r/t*100:0}}),t=((await i.B.query(`
      SELECT
        symbol,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses,
        COALESCE(SUM(profit_loss), 0) as total_pnl,
        COALESCE(AVG(profit_loss), 0) as avg_pnl,
        COALESCE(MAX(profit_loss), 0) as best_trade,
        COALESCE(MIN(profit_loss), 0) as worst_trade
      FROM trade_log
      WHERE timestamp >= $1
      GROUP BY symbol
      ORDER BY total_pnl DESC
      `,[s])).rows??[]).map(e=>{let t=Number(e.total_trades??0),r=Number(e.wins??0);return{...e,total_trades:t,wins:r,losses:Number(e.losses??0),total_pnl:Number(e.total_pnl??0),avg_pnl:Number(e.avg_pnl??0),best_trade:Number(e.best_trade??0),worst_trade:Number(e.worst_trade??0),win_rate:t>0?r/t*100:0}});return l.NextResponse.json({strategies:e,pairs:t})}catch(e){return console.error("Failed to load forex analysis",e),l.NextResponse.json({error:"Failed to load forex analysis"},{status:500})}}let _=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/forex/analysis/route",pathname:"/api/forex/analysis",filename:"route",bundlePath:"app/api/forex/analysis/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/analysis/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:u,staticGenerationAsyncStorage:E,serverHooks:N}=_,m="/api/forex/analysis/route";function C(){return(0,n.patchFetch)({serverHooks:N,staticGenerationAsyncStorage:E})}},1577:(e,t,r)=>{r.d(t,{B:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let s=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:s,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),s=t.X(0,[5822,9967],()=>r(7518));module.exports=s})();