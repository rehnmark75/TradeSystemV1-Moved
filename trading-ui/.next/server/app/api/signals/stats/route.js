"use strict";(()=>{var a={};a.id=1558,a.ids=[1558],a.modules={399:a=>{a.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:a=>{a.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},690:(a,e,s)=>{s.r(e),s.d(e,{originalPathname:()=>g,patchFetch:()=>p,requestAsyncStorage:()=>_,routeModule:()=>l,serverHooks:()=>T,staticGenerationAsyncStorage:()=>d});var t={};s.r(t),s.d(t,{GET:()=>u,dynamic:()=>E});var n=s(7599),r=s(4294),i=s(4588),o=s(2921),c=s(740);let E="force-dynamic";async function u(){let a=await c.d.connect();try{let e=`
      SELECT
        COUNT(*) as total_signals,
        COUNT(*) FILTER (WHERE status = 'active') as active_signals,
        COUNT(*) FILTER (WHERE quality_tier IN ('A+', 'A')) as high_quality,
        COUNT(*) FILTER (WHERE DATE(signal_timestamp) = CURRENT_DATE) as today_signals,
        COUNT(*) FILTER (WHERE claude_analyzed_at IS NOT NULL) as claude_analyzed,
        COUNT(*) FILTER (WHERE claude_grade IN ('A+', 'A')) as claude_high_grade,
        COUNT(*) FILTER (WHERE claude_action = 'STRONG BUY') as claude_strong_buys,
        COUNT(*) FILTER (WHERE claude_action = 'BUY') as claude_buys,
        COUNT(*) FILTER (WHERE claude_analyzed_at IS NULL AND status = 'active') as awaiting_analysis
      FROM stock_scanner_signals
    `,s=await a.query(e),t=`
      SELECT
        r.scanner_name,
        COALESCE(s.signal_count, 0) as signal_count,
        COALESCE(s.avg_score, 0) as avg_score,
        COALESCE(s.active_count, 0) as active_count
      FROM stock_signal_scanners r
      LEFT JOIN (
        SELECT
          scanner_name,
          COUNT(*) as signal_count,
          ROUND(AVG(composite_score)::numeric, 1) as avg_score,
          COUNT(*) FILTER (WHERE status = 'active') as active_count
        FROM stock_scanner_signals
        GROUP BY scanner_name
      ) s ON r.scanner_name = s.scanner_name
      WHERE r.is_active = true
      ORDER BY COALESCE(s.signal_count, 0) DESC, r.scanner_name
    `,n=await a.query(t),r=`
      SELECT
        quality_tier,
        COUNT(*) as count
      FROM stock_scanner_signals
      WHERE status = 'active'
      GROUP BY quality_tier
      ORDER BY
        CASE quality_tier
          WHEN 'A+' THEN 1
          WHEN 'A' THEN 2
          WHEN 'B' THEN 3
          WHEN 'C' THEN 4
          WHEN 'D' THEN 5
        END
    `,i=await a.query(r);return o.NextResponse.json({...s.rows[0],by_scanner:n.rows,by_tier:Object.fromEntries(i.rows.map(a=>[a.quality_tier,Number(a.count)]))})}catch(a){return o.NextResponse.json({error:"Failed to load signal stats"},{status:500})}finally{a.release()}}let l=new n.AppRouteRouteModule({definition:{kind:r.x.APP_ROUTE,page:"/api/signals/stats/route",pathname:"/api/signals/stats",filename:"route",bundlePath:"app/api/signals/stats/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/signals/stats/route.ts",nextConfigOutput:"standalone",userland:t}),{requestAsyncStorage:_,staticGenerationAsyncStorage:d,serverHooks:T}=l,g="/api/signals/stats/route";function p(){return(0,i.patchFetch)({serverHooks:T,staticGenerationAsyncStorage:d})}},740:(a,e,s)=>{s.d(e,{d:()=>n}),function(){var a=Error("Cannot find module 'pg'");throw a.code="MODULE_NOT_FOUND",a}();let t=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",n=Object(function(){var a=Error("Cannot find module 'pg'");throw a.code="MODULE_NOT_FOUND",a}())({connectionString:t,max:10})}};var e=require("../../../../webpack-runtime.js");e.C(a);var s=a=>e(e.s=a),t=e.X(0,[5822,9967],()=>s(690));module.exports=t})();