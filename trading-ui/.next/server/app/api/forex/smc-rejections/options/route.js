"use strict";(()=>{var e={};e.id=5905,e.ids=[5905],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},2467:(e,s,t)=>{t.r(s),t.d(s,{originalPathname:()=>_,patchFetch:()=>j,requestAsyncStorage:()=>m,routeModule:()=>u,serverHooks:()=>E,staticGenerationAsyncStorage:()=>d});var o={};t.r(o),t.d(o,{GET:()=>l,dynamic:()=>c});var r=t(7599),n=t(4294),i=t(4588),a=t(2921),p=t(1577);let c="force-dynamic";async function l(){try{let e=(await p.B.query(`
      SELECT
        COALESCE(
          (SELECT json_agg(DISTINCT rejection_stage ORDER BY rejection_stage)
           FROM smc_simple_rejections),
          '[]'::json
        ) as stages,
        COALESCE(
          (SELECT json_agg(DISTINCT pair ORDER BY pair)
           FROM smc_simple_rejections
           WHERE pair IS NOT NULL),
          '[]'::json
        ) as pairs,
        COALESCE(
          (SELECT json_agg(DISTINCT market_session ORDER BY market_session)
           FROM smc_simple_rejections
           WHERE market_session IS NOT NULL),
          '[]'::json
        ) as sessions,
        EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_name = 'smc_simple_rejections'
        ) as table_exists
      `)).rows[0];if(!e||!e.table_exists)return a.NextResponse.json({stages:["All"],pairs:["All"],sessions:["All"],table_exists:!1});return a.NextResponse.json({stages:["All",...e.stages??[]],pairs:["All",...e.pairs??[]],sessions:["All",...e.sessions??[]],table_exists:!0})}catch(e){return console.error("Failed to load SMC rejection options",e),a.NextResponse.json({error:"Failed to load SMC rejection options"},{status:500})}}let u=new r.AppRouteRouteModule({definition:{kind:n.x.APP_ROUTE,page:"/api/forex/smc-rejections/options/route",pathname:"/api/forex/smc-rejections/options",filename:"route",bundlePath:"app/api/forex/smc-rejections/options/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/smc-rejections/options/route.ts",nextConfigOutput:"standalone",userland:o}),{requestAsyncStorage:m,staticGenerationAsyncStorage:d,serverHooks:E}=u,_="/api/forex/smc-rejections/options/route";function j(){return(0,i.patchFetch)({serverHooks:E,staticGenerationAsyncStorage:d})}},1577:(e,s,t)=>{t.d(s,{B:()=>r}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let o=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",r=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:o,max:10})}};var s=require("../../../../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),o=s.X(0,[5822,9967],()=>t(2467));module.exports=o})();