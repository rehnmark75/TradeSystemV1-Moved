"use strict";(()=>{var e={};e.id=1534,e.ids=[1534],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},8288:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>x,patchFetch:()=>g,requestAsyncStorage:()=>c,routeModule:()=>u,serverHooks:()=>f,staticGenerationAsyncStorage:()=>m});var o={};t.r(o),t.d(o,{GET:()=>p,dynamic:()=>d});var s=t(7599),a=t(4294),n=t(4588),i=t(2921),l=t(1577);let d="force-dynamic";async function p(e){let{searchParams:r}=new URL(e.url),t=function(e){if(!e)return 100;let r=Number(e);return!Number.isFinite(r)||r<=0?100:r}(r.get("limit"));try{let e=((await l.B.query(`
      SELECT
        id,
        symbol,
        direction,
        timestamp,
        status,
        profit_loss,
        pnl_currency
      FROM trade_log
      WHERE status IN ('closed', 'tracking')
      ORDER BY timestamp DESC
      LIMIT $1
      `,[t])).rows??[]).map(e=>({...e,profit_loss:null==e.profit_loss?null:Number(e.profit_loss),pnl_display:null==e.profit_loss?"Open":`${e.profit_loss>=0?"+":""}${Number(e.profit_loss).toFixed(2)} ${e.pnl_currency??""}`.trim()}));return i.NextResponse.json({trades:e})}catch(e){return console.error("Failed to load trade list",e),i.NextResponse.json({error:"Failed to load trade list"},{status:500})}}let u=new s.AppRouteRouteModule({definition:{kind:a.x.APP_ROUTE,page:"/api/forex/trade-analysis/trades/route",pathname:"/api/forex/trade-analysis/trades",filename:"route",bundlePath:"app/api/forex/trade-analysis/trades/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/trade-analysis/trades/route.ts",nextConfigOutput:"standalone",userland:o}),{requestAsyncStorage:c,staticGenerationAsyncStorage:m,serverHooks:f}=u,x="/api/forex/trade-analysis/trades/route";function g(){return(0,n.patchFetch)({serverHooks:f,staticGenerationAsyncStorage:m})}},1577:(e,r,t)=>{t.d(r,{B:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let o=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:o,max:10})}};var r=require("../../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),o=r.X(0,[5822,9967],()=>t(8288));module.exports=o})();