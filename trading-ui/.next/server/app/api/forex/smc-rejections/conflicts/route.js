"use strict";(()=>{var e={};e.id=9642,e.ids=[9642],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4282:(e,t,s)=>{s.r(t),s.d(t,{originalPathname:()=>N,patchFetch:()=>E,requestAsyncStorage:()=>m,routeModule:()=>u,serverHooks:()=>l,staticGenerationAsyncStorage:()=>d});var r={};s.r(r),s.d(r,{GET:()=>_,dynamic:()=>p});var o=s(7599),a=s(4294),i=s(4588),n=s(2921),c=s(1577);let p="force-dynamic";async function _(e){let{searchParams:t}=new URL(e.url),s=function(e){if(!e)return 7;let t=Number(e);return!Number.isFinite(t)||t<=0?7:t}(t.get("days"));try{let e=await c.B.query(`
      SELECT
        COUNT(*) as total,
        COUNT(DISTINCT epic) as unique_pairs,
        COUNT(DISTINCT market_session) as sessions_affected
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${s} days'
        AND rejection_stage = 'SMC_CONFLICT'
      `),t=await c.B.query(`
      SELECT pair, COUNT(*) as count
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${s} days'
        AND rejection_stage = 'SMC_CONFLICT'
      GROUP BY pair
      ORDER BY count DESC
      `),r=await c.B.query(`
      SELECT market_session, COUNT(*) as count
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${s} days'
        AND rejection_stage = 'SMC_CONFLICT'
        AND market_session IS NOT NULL
      GROUP BY market_session
      ORDER BY count DESC
      `),o=await c.B.query(`
      SELECT rejection_reason, COUNT(*) as count
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${s} days'
        AND rejection_stage = 'SMC_CONFLICT'
      GROUP BY rejection_reason
      ORDER BY count DESC
      LIMIT 10
      `),a=await c.B.query(`
      SELECT
        id,
        scan_timestamp,
        epic,
        pair,
        rejection_reason,
        attempted_direction,
        current_price,
        market_hour,
        market_session,
        potential_entry,
        potential_stop_loss,
        potential_take_profit,
        potential_risk_pips,
        potential_reward_pips,
        potential_rr_ratio,
        confidence_score,
        rejection_details
      FROM smc_simple_rejections
      WHERE scan_timestamp >= NOW() - INTERVAL '${s} days'
        AND rejection_stage = 'SMC_CONFLICT'
      ORDER BY scan_timestamp DESC
      LIMIT 500
      `),i=e.rows[0]??{};return n.NextResponse.json({stats:{total:Number(i.total??0),unique_pairs:Number(i.unique_pairs??0),sessions_affected:Number(i.sessions_affected??0)},by_pair:t.rows??[],by_session:r.rows??[],top_reasons:o.rows??[],details:a.rows??[]})}catch(e){return console.error("Failed to load SMC conflict data",e),n.NextResponse.json({error:"Failed to load SMC conflict data"},{status:500})}}let u=new o.AppRouteRouteModule({definition:{kind:a.x.APP_ROUTE,page:"/api/forex/smc-rejections/conflicts/route",pathname:"/api/forex/smc-rejections/conflicts",filename:"route",bundlePath:"app/api/forex/smc-rejections/conflicts/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/smc-rejections/conflicts/route.ts",nextConfigOutput:"standalone",userland:r}),{requestAsyncStorage:m,staticGenerationAsyncStorage:d,serverHooks:l}=u,N="/api/forex/smc-rejections/conflicts/route";function E(){return(0,i.patchFetch)({serverHooks:l,staticGenerationAsyncStorage:d})}},1577:(e,t,s)=>{s.d(t,{B:()=>o}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let r=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",o=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:r,max:10})}};var t=require("../../../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),r=t.X(0,[5822,9967],()=>s(4282));module.exports=r})();