"use strict";(()=>{var e={};e.id=9946,e.ids=[9946],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},8407:(e,t,i)=>{i.r(t),i.d(t,{originalPathname:()=>v,patchFetch:()=>y,requestAsyncStorage:()=>d,routeModule:()=>u,serverHooks:()=>h,staticGenerationAsyncStorage:()=>p});var a={};i.r(a),i.d(a,{GET:()=>_,dynamic:()=>m});var n=i(7599),r=i(4294),s=i(4588),l=i(2921),o=i(1577);let m="force-dynamic";function c(e){if(!e)return null;let t=new Date(e);return Number.isNaN(t.valueOf())?null:t}async function g(){return((await o.B.query(`
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'market_intelligence_history'
    `)).rows??[]).map(e=>e.column_name)}async function _(e){let{searchParams:t}=new URL(e.url),i=(t.get("source")||"comprehensive").toLowerCase(),{start:a,end:n}=function(e,t){let i=t??new Date;return{start:e??new Date(new Date(i).setDate(i.getDate()-7)),end:i}}(c(t.get("start")),c(t.get("end")));try{let e=[],t=[];if("comprehensive"===i||"both"===i){let t=await g();if(t.length){let i=[];t.includes("individual_epic_regimes")&&i.push("mih.individual_epic_regimes"),t.includes("pair_analyses")&&i.push("mih.pair_analyses");let r=`
          SELECT
            ${["mih.id","mih.scan_timestamp","mih.scan_cycle_id","mih.epic_list","mih.epic_count","mih.dominant_regime as regime","mih.regime_confidence","mih.current_session as session","mih.session_volatility as volatility_level","mih.market_bias","mih.average_trend_strength","mih.average_volatility","mih.risk_sentiment","mih.recommended_strategy","mih.confidence_threshold","mih.intelligence_source","mih.regime_trending_score","mih.regime_ranging_score","mih.regime_breakout_score","mih.regime_reversal_score","mih.regime_high_vol_score","mih.regime_low_vol_score",...i].join(", ")}
          FROM market_intelligence_history mih
          WHERE mih.scan_timestamp >= $1
            AND mih.scan_timestamp <= $2
          ORDER BY mih.scan_timestamp DESC
        `;e=(await o.B.query(r,[a,n])).rows??[]}}("signal"===i||"both"===i)&&(t=(await o.B.query(`
        SELECT
          a.id,
          a.alert_timestamp,
          a.epic,
          a.strategy,
          a.signal_type,
          a.confidence_score,
          a.strategy_metadata,
          (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'dominant_regime') as regime,
          (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'confidence')::float as regime_confidence,
          (a.strategy_metadata::json->'market_intelligence'->'session_analysis'->>'current_session') as session,
          (a.strategy_metadata::json->'market_intelligence'->>'volatility_level') as volatility_level,
          (a.strategy_metadata::json->'market_intelligence'->>'intelligence_source') as intelligence_source
        FROM alert_history a
        WHERE a.alert_timestamp >= $1
          AND a.alert_timestamp <= $2
          AND a.strategy_metadata IS NOT NULL
          AND (a.strategy_metadata::json->'market_intelligence') IS NOT NULL
        ORDER BY a.alert_timestamp DESC
        `,[a,n])).rows??[]);let r="signal"===i?t:e.length?e:t,s={total:r.length,avg_epics:r.reduce((e,t)=>{let i=Number(t.epic_count??0);return e+(Number.isFinite(i)?i:0)},0),unique_regimes:new Set(r.map(e=>e.regime).filter(Boolean)).size,avg_confidence:r.length?r.reduce((e,t)=>{let i=Number(t.regime_confidence??0);return e+(Number.isFinite(i)?i:0)},0)/r.length:0};s.avg_epics=r.length?s.avg_epics/r.length:0;let m={},c={},_={},u={};for(let e of r)e.regime&&(m[String(e.regime)]=(m[String(e.regime)]||0)+1),e.session&&(c[String(e.session)]=(c[String(e.session)]||0)+1),e.volatility_level&&(_[String(e.volatility_level)]=(_[String(e.volatility_level)]||0)+1),e.intelligence_source&&(u[String(e.intelligence_source)]=(u[String(e.intelligence_source)]||0)+1);return l.NextResponse.json({range:{start:a,end:n},source:i,summary:s,regimes:m,sessions:c,volatility:_,intelligence_sources:u,comprehensive:e,signals:t})}catch(e){return console.error("Failed to load market intelligence",e),l.NextResponse.json({error:"Failed to load market intelligence"},{status:500})}}let u=new n.AppRouteRouteModule({definition:{kind:r.x.APP_ROUTE,page:"/api/forex/market-intelligence/route",pathname:"/api/forex/market-intelligence",filename:"route",bundlePath:"app/api/forex/market-intelligence/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/market-intelligence/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:d,staticGenerationAsyncStorage:p,serverHooks:h}=u,v="/api/forex/market-intelligence/route";function y(){return(0,s.patchFetch)({serverHooks:h,staticGenerationAsyncStorage:p})}},1577:(e,t,i)=>{i.d(t,{B:()=>n}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",n=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var i=e=>t(t.s=e),a=t.X(0,[5822,9967],()=>i(8407));module.exports=a})();