"use strict";(()=>{var e={};e.id=2650,e.ids=[2650],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},3916:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>g,patchFetch:()=>h,requestAsyncStorage:()=>u,routeModule:()=>d,serverHooks:()=>_,staticGenerationAsyncStorage:()=>l});var s={};t.r(s),t.d(s,{GET:()=>p,dynamic:()=>m});var a=t(7599),n=t(4294),o=t(4588),i=t(2921),c=t(740);let m="force-dynamic";async function p(e){let{searchParams:r}=new URL(e.url),t=Number(r.get("minRs")||80),s=Number(r.get("limit")||30),a=await c.d.connect();try{let e=`
      SELECT
        m.ticker,
        i.name,
        i.sector,
        m.current_price,
        m.rs_vs_spy,
        m.rs_percentile,
        m.rs_trend,
        m.price_change_20d,
        m.trend_strength,
        m.ma_alignment,
        m.atr_percent,
        m.rsi_14,
        m.pct_from_52w_high
      FROM stock_screening_metrics m
      JOIN stock_instruments i ON m.ticker = i.ticker
      WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        AND m.rs_percentile >= $1
        AND m.rs_percentile IS NOT NULL
      ORDER BY m.rs_percentile DESC
      LIMIT $2
    `,r=await a.query(e,[t,s]);return i.NextResponse.json({rows:r.rows||[]})}catch(e){return i.NextResponse.json({error:"Failed to load RS leaders"},{status:500})}finally{a.release()}}let d=new a.AppRouteRouteModule({definition:{kind:n.x.APP_ROUTE,page:"/api/market/rs-leaders/route",pathname:"/api/market/rs-leaders",filename:"route",bundlePath:"app/api/market/rs-leaders/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/market/rs-leaders/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:u,staticGenerationAsyncStorage:l,serverHooks:_}=d,g="/api/market/rs-leaders/route";function h(){return(0,o.patchFetch)({serverHooks:_,staticGenerationAsyncStorage:l})}},740:(e,r,t)=>{t.d(r,{d:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let s=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:s,max:10})}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),s=r.X(0,[5822,9967],()=>t(3916));module.exports=s})();