"use strict";(()=>{var e={};e.id=3165,e.ids=[3165],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},2218:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>_,patchFetch:()=>h,requestAsyncStorage:()=>g,routeModule:()=>c,serverHooks:()=>m,staticGenerationAsyncStorage:()=>l});var a={};r.r(a),r.d(a,{GET:()=>p,dynamic:()=>u});var s=r(7599),o=r(4294),i=r(4588),n=r(2921),d=r(4372);let u="force-dynamic";async function p(e){let{searchParams:t}=new URL(e.url),r=Number(t.get("limit")??50);try{let e=await d.A.query(`
        SELECT
          id,
          config_id,
          pair_override_id,
          change_type,
          changed_by,
          changed_at,
          change_reason,
          previous_values,
          new_values
        FROM smc_simple_config_audit
        ORDER BY changed_at DESC
        LIMIT $1
      `,[r]);return n.NextResponse.json(e.rows??[])}catch(e){return console.error("Failed to load SMC audit history",e),n.NextResponse.json({error:"Failed to load SMC audit history"},{status:500})}}let c=new s.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/settings/strategy/smc/audit/route",pathname:"/api/settings/strategy/smc/audit",filename:"route",bundlePath:"app/api/settings/strategy/smc/audit/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/settings/strategy/smc/audit/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:g,staticGenerationAsyncStorage:l,serverHooks:m}=c,_="/api/settings/strategy/smc/audit/route";function h(){return(0,i.patchFetch)({serverHooks:m,staticGenerationAsyncStorage:l})}},4372:(e,t,r)=>{r.d(t,{A:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.STRATEGY_CONFIG_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/strategy_config",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var t=require("../../../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),a=t.X(0,[5822,9967],()=>r(2218));module.exports=a})();