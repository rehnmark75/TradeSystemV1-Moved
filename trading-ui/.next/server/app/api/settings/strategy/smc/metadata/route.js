"use strict";(()=>{var e={};e.id=8461,e.ids=[8461],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},8471:(e,t,a)=>{a.r(t),a.d(t,{originalPathname:()=>_,patchFetch:()=>y,requestAsyncStorage:()=>m,routeModule:()=>c,serverHooks:()=>g,staticGenerationAsyncStorage:()=>l});var r={};a.r(r),a.d(r,{GET:()=>u,dynamic:()=>p});var s=a(7599),o=a(4294),n=a(4588),i=a(2921),d=a(4372);let p="force-dynamic";async function u(){try{let e=await d.A.query(`
        SELECT
          id,
          parameter_name,
          display_name,
          category,
          subcategory,
          data_type,
          min_value,
          max_value,
          default_value,
          description,
          help_text,
          display_order,
          is_advanced,
          requires_restart,
          valid_options,
          unit
        FROM smc_simple_parameter_metadata
        ORDER BY category, display_order, display_name
      `);return i.NextResponse.json(e.rows??[])}catch(e){return console.error("Failed to load SMC metadata",e),i.NextResponse.json({error:"Failed to load SMC metadata"},{status:500})}}let c=new s.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/settings/strategy/smc/metadata/route",pathname:"/api/settings/strategy/smc/metadata",filename:"route",bundlePath:"app/api/settings/strategy/smc/metadata/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/settings/strategy/smc/metadata/route.ts",nextConfigOutput:"standalone",userland:r}),{requestAsyncStorage:m,staticGenerationAsyncStorage:l,serverHooks:g}=c,_="/api/settings/strategy/smc/metadata/route";function y(){return(0,n.patchFetch)({serverHooks:g,staticGenerationAsyncStorage:l})}},4372:(e,t,a)=>{a.d(t,{A:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let r=process.env.STRATEGY_CONFIG_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/strategy_config",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:r,max:10})}};var t=require("../../../../../../webpack-runtime.js");t.C(e);var a=e=>t(t.s=e),r=t.X(0,[5822,9967],()=>a(8471));module.exports=r})();