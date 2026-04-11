"use strict";(()=>{var e={};e.id=6946,e.ids=[6946],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},2527:(e,t,s)=>{s.r(t),s.d(t,{originalPathname:()=>O,patchFetch:()=>l,requestAsyncStorage:()=>d,routeModule:()=>p,serverHooks:()=>E,staticGenerationAsyncStorage:()=>m});var r={};s.r(r),s.d(r,{GET:()=>u,dynamic:()=>_});var a=s(7599),o=s(4294),n=s(4588),i=s(2921),c=s(1577);let _="force-dynamic";async function u(e){let{searchParams:t}=new URL(e.url),s=function(e){if(!e)return 7;let t=Number(e);return!Number.isFinite(t)||t<=0?7:t}(t.get("days"));try{let e=(await c.B.query(`
      WITH base_data AS (
        SELECT
          epic,
          pair,
          rejection_stage,
          confidence_score,
          attempted_direction
        FROM smc_simple_rejections
        WHERE scan_timestamp >= NOW() - INTERVAL '${s} days'
      ),
      stage_counts AS (
        SELECT
          rejection_stage,
          COUNT(*) as stage_count
        FROM base_data
        GROUP BY rejection_stage
      ),
      direction_counts AS (
        SELECT
          attempted_direction,
          COUNT(*) as dir_count
        FROM base_data
        WHERE attempted_direction IS NOT NULL
        GROUP BY attempted_direction
      ),
      totals AS (
        SELECT
          COUNT(*) as total,
          COUNT(DISTINCT epic) as unique_pairs
        FROM base_data
      ),
      near_misses AS (
        SELECT COUNT(*) as near_miss_count
        FROM base_data
        WHERE rejection_stage = 'CONFIDENCE'
          AND confidence_score >= 0.45
      ),
      smc_conflicts AS (
        SELECT COUNT(*) as conflict_count
        FROM base_data
        WHERE rejection_stage = 'SMC_CONFLICT'
      ),
      top_pair AS (
        SELECT pair, COUNT(*) as pair_count
        FROM base_data
        WHERE pair IS NOT NULL
        GROUP BY pair
        ORDER BY pair_count DESC
        LIMIT 1
      )
      SELECT
        t.total,
        t.unique_pairs,
        nm.near_miss_count,
        sc.conflict_count,
        tp.pair as most_rejected_pair,
        (SELECT json_object_agg(rejection_stage, stage_count) FROM stage_counts) as by_stage,
        (SELECT json_object_agg(attempted_direction, dir_count) FROM direction_counts) as by_direction
      FROM totals t
      CROSS JOIN near_misses nm
      CROSS JOIN smc_conflicts sc
      LEFT JOIN top_pair tp ON true
      `)).rows[0];return i.NextResponse.json({total:Number(e?.total??0),unique_pairs:Number(e?.unique_pairs??0),near_misses:Number(e?.near_miss_count??0),smc_conflicts:Number(e?.conflict_count??0),most_rejected_pair:e?.most_rejected_pair??"N/A",by_stage:e?.by_stage??{},by_direction:e?.by_direction??{}})}catch(e){return console.error("Failed to load SMC rejection stats",e),i.NextResponse.json({error:"Failed to load SMC rejection stats"},{status:500})}}let p=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/forex/smc-rejections/stats/route",pathname:"/api/forex/smc-rejections/stats",filename:"route",bundlePath:"app/api/forex/smc-rejections/stats/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/smc-rejections/stats/route.ts",nextConfigOutput:"standalone",userland:r}),{requestAsyncStorage:d,staticGenerationAsyncStorage:m,serverHooks:E}=p,O="/api/forex/smc-rejections/stats/route";function l(){return(0,n.patchFetch)({serverHooks:E,staticGenerationAsyncStorage:m})}},1577:(e,t,s)=>{s.d(t,{B:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let r=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:r,max:10})}};var t=require("../../../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),r=t.X(0,[5822,9967],()=>s(2527));module.exports=r})();