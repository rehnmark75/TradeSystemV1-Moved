"use strict";(()=>{var e={};e.id=6725,e.ids=[6725],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},751:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>v,patchFetch:()=>b,requestAsyncStorage:()=>u,routeModule:()=>p,serverHooks:()=>g,staticGenerationAsyncStorage:()=>d});var a={};r.r(a),r.d(a,{GET:()=>m,dynamic:()=>c});var n=r(7599),o=r(4294),s=r(4588),i=r(2921),_=r(740);let c="force-dynamic",l=e=>{let t=Number(e);return Number.isNaN(t)?0:t};async function m(){let e=await _.d.connect();try{let t=`
      SELECT
        calculation_date,
        market_regime,
        spy_price,
        spy_sma50,
        spy_sma200,
        spy_vs_sma50_pct,
        spy_vs_sma200_pct,
        spy_trend,
        pct_above_sma200,
        pct_above_sma50,
        pct_above_sma20,
        new_highs_count,
        new_lows_count,
        high_low_ratio,
        advancing_count,
        declining_count,
        ad_ratio,
        avg_atr_pct,
        volatility_regime,
        recommended_strategies
      FROM market_context
      ORDER BY calculation_date DESC
      LIMIT 1
    `,r=await e.query(t);if(r.rows.length){let e=r.rows[0];if(e.recommended_strategies&&"string"==typeof e.recommended_strategies)try{e.recommended_strategies=JSON.parse(e.recommended_strategies)}catch(t){e.recommended_strategies=null}return i.NextResponse.json({row:e})}let a=`
      SELECT
        COUNT(*) FILTER (WHERE current_price > sma_200) as above_200,
        COUNT(*) FILTER (WHERE current_price > sma_50) as above_50,
        COUNT(*) FILTER (WHERE current_price > sma_20) as above_20,
        COUNT(*) FILTER (WHERE trend_strength IN ('strong_up', 'up')) as bullish,
        COUNT(*) FILTER (WHERE trend_strength IN ('strong_down', 'down')) as bearish,
        COUNT(*) as total,
        AVG(atr_percent) as avg_atr,
        AVG(current_price) as avg_price
      FROM stock_screening_metrics
      WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
    `,n=(await e.query(a)).rows[0];if(!n)return i.NextResponse.json({row:null});let o=l(n.total)||1,s=l(n.above_200)/o*100,_=l(n.above_50)/o*100,c=l(n.above_20)/o*100,m=l(n.bullish),p=l(n.bearish),u=p>0?m/p:m,d=l(n.avg_atr),g="normal";g=d<2?"low":d<4?"normal":d<6?"high":"extreme";let v="bear_confirmed",b="falling";s>60&&_>50?(v="bull_confirmed",b="rising"):s>50?(v="bull_weakening",b="flat"):s>40&&(v="bear_weakening",b="flat");let E=l(n.avg_price),h=E?Number((5*E).toFixed(2)):0,w="bull_confirmed"===v?{trend_following:.8,breakout:.7,pullback:.6,mean_reversion:.2}:"bull_weakening"===v?{trend_following:.5,breakout:.4,pullback:.7,mean_reversion:.4}:"bear_weakening"===v?{trend_following:.3,breakout:.3,pullback:.5,mean_reversion:.6}:{trend_following:.2,breakout:.2,pullback:.3,mean_reversion:.7};return i.NextResponse.json({row:{market_regime:v,spy_price:h,spy_sma50:h*(_>50?.98:1.02),spy_sma200:h*(s>50?.95:1.05),spy_vs_sma50_pct:_-50,spy_vs_sma200_pct:s-50,spy_trend:b,pct_above_sma200:s,pct_above_sma50:_,pct_above_sma20:c,new_highs_count:0,new_lows_count:0,high_low_ratio:1,advancing_count:m,declining_count:p,ad_ratio:u,avg_atr_pct:d,volatility_regime:g,recommended_strategies:w}})}catch(e){return i.NextResponse.json({error:"Failed to load market regime"},{status:500})}finally{e.release()}}let p=new n.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/market/regime/route",pathname:"/api/market/regime",filename:"route",bundlePath:"app/api/market/regime/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/market/regime/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:u,staticGenerationAsyncStorage:d,serverHooks:g}=p,v="/api/market/regime/route";function b(){return(0,s.patchFetch)({serverHooks:g,staticGenerationAsyncStorage:d})}},740:(e,t,r)=>{r.d(t,{d:()=>n}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",n=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),a=t.X(0,[5822,9967],()=>r(751));module.exports=a})();