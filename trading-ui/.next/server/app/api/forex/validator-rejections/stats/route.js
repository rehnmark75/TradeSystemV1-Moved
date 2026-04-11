"use strict";(()=>{var t={};t.id=538,t.ids=[538],t.modules={399:t=>{t.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:t=>{t.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},7218:(t,e,a)=>{a.r(e),a.d(e,{originalPathname:()=>y,patchFetch:()=>S,requestAsyncStorage:()=>c,routeModule:()=>u,serverHooks:()=>E,staticGenerationAsyncStorage:()=>d});var r={};a.r(r),a.d(r,{GET:()=>_,dynamic:()=>l});var o=a(7599),s=a(4294),p=a(4588),n=a(2921),i=a(1577);let l="force-dynamic";async function _(t){let{searchParams:e}=new URL(t.url),a=function(t){if(!t)return 7;let e=Number(t);return!Number.isFinite(e)||e<=0?7:e}(e.get("days"));try{let t=(await i.B.query(`
      WITH base AS (
        SELECT step, pair, signal_type, confidence_score, rr_ratio,
               lpf_penalty, lpf_triggered_rules, created_at
        FROM validator_rejections
        WHERE created_at >= NOW() - INTERVAL '${a} days'
      ),
      totals AS (
        SELECT COUNT(*) AS total, COUNT(DISTINCT pair) AS unique_pairs
        FROM base
      ),
      by_step AS (
        SELECT step, COUNT(*) AS cnt
        FROM base GROUP BY step
      ),
      by_pair AS (
        SELECT pair, COUNT(*) AS cnt
        FROM base WHERE pair IS NOT NULL
        GROUP BY pair ORDER BY cnt DESC LIMIT 10
      ),
      by_direction AS (
        SELECT signal_type, COUNT(*) AS cnt
        FROM base WHERE signal_type IS NOT NULL
        GROUP BY signal_type
      ),
      top_step AS (
        SELECT step FROM by_step ORDER BY cnt DESC LIMIT 1
      ),
      top_pair AS (
        SELECT pair FROM by_pair LIMIT 1
      ),
      lpf_stats AS (
        SELECT
          COUNT(*) AS total_lpf,
          AVG(lpf_penalty) AS avg_penalty,
          MAX(lpf_penalty) AS max_penalty
        FROM base WHERE step = 'LPF'
      )
      SELECT
        t.total,
        t.unique_pairs,
        ts.step AS top_step,
        tp.pair AS top_pair,
        ls.total_lpf,
        ls.avg_penalty,
        ls.max_penalty,
        (SELECT json_object_agg(step, cnt) FROM by_step) AS by_step,
        (SELECT json_agg(json_build_object('pair', pair, 'count', cnt)) FROM by_pair) AS by_pair,
        (SELECT json_object_agg(signal_type, cnt) FROM by_direction) AS by_direction
      FROM totals t
      CROSS JOIN lpf_stats ls
      LEFT JOIN top_step ts ON true
      LEFT JOIN top_pair tp ON true
    `)).rows[0];return n.NextResponse.json({total:Number(t?.total??0),unique_pairs:Number(t?.unique_pairs??0),top_step:t?.top_step??"N/A",top_pair:t?.top_pair??"N/A",total_lpf:Number(t?.total_lpf??0),avg_lpf_penalty:t?.avg_penalty?Number(t.avg_penalty).toFixed(2):null,max_lpf_penalty:t?.max_penalty?Number(t.max_penalty).toFixed(2):null,by_step:t?.by_step??{},by_pair:t?.by_pair??[],by_direction:t?.by_direction??{}})}catch(t){return console.error("validator-rejections/stats error:",t),n.NextResponse.json({error:"Failed to load stats"},{status:500})}}let u=new o.AppRouteRouteModule({definition:{kind:s.x.APP_ROUTE,page:"/api/forex/validator-rejections/stats/route",pathname:"/api/forex/validator-rejections/stats",filename:"route",bundlePath:"app/api/forex/validator-rejections/stats/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/validator-rejections/stats/route.ts",nextConfigOutput:"standalone",userland:r}),{requestAsyncStorage:c,staticGenerationAsyncStorage:d,serverHooks:E}=u,y="/api/forex/validator-rejections/stats/route";function S(){return(0,p.patchFetch)({serverHooks:E,staticGenerationAsyncStorage:d})}},1577:(t,e,a)=>{a.d(e,{B:()=>o}),function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}();let r=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",o=Object(function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}())({connectionString:r,max:10})}};var e=require("../../../../../webpack-runtime.js");e.C(t);var a=t=>e(e.s=t),r=e.X(0,[5822,9967],()=>a(7218));module.exports=r})();