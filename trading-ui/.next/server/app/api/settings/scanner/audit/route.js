"use strict";(()=>{var e={};e.id=8733,e.ids=[8733],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},3676:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>m,patchFetch:()=>_,requestAsyncStorage:()=>g,routeModule:()=>p,serverHooks:()=>h,staticGenerationAsyncStorage:()=>l});var n={};r.r(n),r.d(n,{GET:()=>c,dynamic:()=>d});var a=r(7599),o=r(4294),s=r(4588),i=r(2921),u=r(4372);let d="force-dynamic";async function c(e){let{searchParams:t}=new URL(e.url),r=Number(t.get("limit")??50),n=t.get("category");try{let e=await u.A.query(`
        SELECT
          id,
          config_id,
          change_type,
          changed_by,
          changed_at,
          change_reason,
          previous_values,
          new_values,
          category
        FROM scanner_config_audit
        ${n?"WHERE category = $1":""}
        ORDER BY changed_at DESC
        LIMIT ${n?"$2":"$1"}
      `,n?[n,r]:[r]);return i.NextResponse.json(e.rows??[])}catch(e){return console.error("Failed to load scanner audit history",e),i.NextResponse.json({error:"Failed to load scanner audit history"},{status:500})}}let p=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/settings/scanner/audit/route",pathname:"/api/settings/scanner/audit",filename:"route",bundlePath:"app/api/settings/scanner/audit/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/settings/scanner/audit/route.ts",nextConfigOutput:"standalone",userland:n}),{requestAsyncStorage:g,staticGenerationAsyncStorage:l,serverHooks:h}=p,m="/api/settings/scanner/audit/route";function _(){return(0,s.patchFetch)({serverHooks:h,staticGenerationAsyncStorage:l})}},4372:(e,t,r)=>{r.d(t,{A:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let n=process.env.STRATEGY_CONFIG_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/strategy_config",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:n,max:10})}};var t=require("../../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),n=t.X(0,[5822,9967],()=>r(3676));module.exports=n})();