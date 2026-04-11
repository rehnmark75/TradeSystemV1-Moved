"use strict";(()=>{var e={};e.id=6180,e.ids=[6180],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},4895:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>v,patchFetch:()=>O,requestAsyncStorage:()=>c,routeModule:()=>u,serverHooks:()=>f,staticGenerationAsyncStorage:()=>_});var a={};t.r(a),t.d(a,{GET:()=>d,dynamic:()=>p});var o=t(7599),n=t(4294),i=t(4588),s=t(2921),l=t(1577);let p="force-dynamic";async function d(e){let{searchParams:r}=new URL(e.url),t=function(e){if(!e)return 7;let r=Number(e);return!Number.isFinite(r)||r<=0?7:r}(r.get("days"));try{let e=await l.B.query(`
      SELECT
        rule_name,
        COUNT(*) AS times_triggered,
        COUNT(DISTINCT pair) AS pairs_affected,
        AVG(vr.lpf_penalty) AS avg_total_penalty
      FROM validator_rejections vr,
           jsonb_array_elements_text(vr.lpf_triggered_rules) AS rule_name
      WHERE vr.step = 'LPF'
        AND vr.created_at >= NOW() - INTERVAL '${t} days'
        AND vr.lpf_triggered_rules IS NOT NULL
      GROUP BY rule_name
      ORDER BY times_triggered DESC
    `),r=await l.B.query(`
      SELECT
        pair,
        COUNT(*) AS total_lpf_blocks,
        AVG(lpf_penalty) AS avg_penalty,
        MAX(lpf_penalty) AS max_penalty
      FROM validator_rejections
      WHERE step = 'LPF'
        AND created_at >= NOW() - INTERVAL '${t} days'
        AND pair IS NOT NULL
      GROUP BY pair
      ORDER BY total_lpf_blocks DESC
    `),a=await l.B.query(`
      SELECT
        EXTRACT(HOUR FROM created_at) AS hour,
        COUNT(*) AS count
      FROM validator_rejections
      WHERE step = 'LPF'
        AND created_at >= NOW() - INTERVAL '${t} days'
      GROUP BY hour
      ORDER BY hour
    `);return s.NextResponse.json({rule_breakdown:e.rows,by_pair:r.rows,hourly:a.rows})}catch(e){return console.error("validator-rejections/lpf error:",e),s.NextResponse.json({error:"Failed to load LPF detail"},{status:500})}}let u=new o.AppRouteRouteModule({definition:{kind:n.x.APP_ROUTE,page:"/api/forex/validator-rejections/lpf/route",pathname:"/api/forex/validator-rejections/lpf",filename:"route",bundlePath:"app/api/forex/validator-rejections/lpf/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/validator-rejections/lpf/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:c,staticGenerationAsyncStorage:_,serverHooks:f}=u,v="/api/forex/validator-rejections/lpf/route";function O(){return(0,i.patchFetch)({serverHooks:f,staticGenerationAsyncStorage:_})}},1577:(e,r,t)=>{t.d(r,{B:()=>o}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",o=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var r=require("../../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),a=r.X(0,[5822,9967],()=>t(4895));module.exports=a})();