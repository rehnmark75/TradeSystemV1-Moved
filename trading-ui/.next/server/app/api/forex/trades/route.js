"use strict";(()=>{var e={};e.id=1238,e.ids=[1238],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},6858:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>E,patchFetch:()=>x,requestAsyncStorage:()=>_,routeModule:()=>m,serverHooks:()=>g,staticGenerationAsyncStorage:()=>f});var n={};r.r(n),r.d(n,{GET:()=>c,dynamic:()=>l});var o=r(7599),i=r(4294),a=r(4588),s=r(2921),d=r(1577);let l="force-dynamic",p=new Set(["pending","pending_limit"]),u=new Set(["limit_rejected","limit_cancelled"]);async function c(e){let{searchParams:t}=new URL(e.url),r=function(e){if(!e)return 30;let t=Number(e);return!Number.isFinite(t)||t<=0?30:t}(t.get("days")),n=new Date;n.setDate(n.getDate()-r);try{let e=((await d.B.query(`
      SELECT
        t.id,
        t.symbol,
        t.entry_price,
        t.direction,
        t.timestamp,
        t.status,
        t.profit_loss,
        t.pnl_currency,
        a.strategy
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      ORDER BY t.timestamp DESC
      `,[n])).rows??[]).map(e=>{var t;let r=e.status??"pending",n=null==e.profit_loss?null:Number(e.profit_loss);return{...e,profit_loss:n,trade_result:null!=n?n>0?"WIN":n<0?"LOSS":"BREAKEVEN":p.has(r)?"PENDING":"tracking"===r?"OPEN":"limit_not_filled"===r?"EXPIRED":u.has(r)?"REJECTED":"PENDING",profit_loss_formatted:(t=e.pnl_currency,null==n?({tracking:"Open",limit_not_filled:"Not Filled",limit_rejected:"Rejected",limit_cancelled:"Cancelled",pending:"Pending",pending_limit:"Pending"})[r]??"Pending":`${n>0?"+":""}${n.toFixed(2)} ${t??""}`.trim())}});return s.NextResponse.json({trades:e})}catch(e){return console.error("Failed to load forex trades",e),s.NextResponse.json({error:"Failed to load forex trades"},{status:500})}}let m=new o.AppRouteRouteModule({definition:{kind:i.x.APP_ROUTE,page:"/api/forex/trades/route",pathname:"/api/forex/trades",filename:"route",bundlePath:"app/api/forex/trades/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/trades/route.ts",nextConfigOutput:"standalone",userland:n}),{requestAsyncStorage:_,staticGenerationAsyncStorage:f,serverHooks:g}=m,E="/api/forex/trades/route";function x(){return(0,a.patchFetch)({serverHooks:g,staticGenerationAsyncStorage:f})}},1577:(e,t,r)=>{r.d(t,{B:()=>o}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let n=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",o=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:n,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),n=t.X(0,[5822,9967],()=>r(6858));module.exports=n})();