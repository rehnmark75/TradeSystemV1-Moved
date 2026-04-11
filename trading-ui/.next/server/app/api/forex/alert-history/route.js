"use strict";(()=>{var e={};e.id=282,e.ids=[282],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},6142:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>m,patchFetch:()=>g,requestAsyncStorage:()=>c,routeModule:()=>d,serverHooks:()=>E,staticGenerationAsyncStorage:()=>_});var a={};t.r(a),t.d(a,{GET:()=>p,dynamic:()=>n});var s=t(7599),o=t(4294),i=t(4588),l=t(2921),u=t(1577);let n="force-dynamic";async function p(e){let{searchParams:r}=new URL(e.url),t=function(e){if(!e)return 1;let r=Number(e);return!Number.isFinite(r)||r<=0?1:r}(r.get("days")),a=function(e){if(!e)return 25;let r=Number(e);return!Number.isFinite(r)||r<=0?25:r}(r.get("limit")),s=function(e){if(!e)return 1;let r=Number(e);return!Number.isFinite(r)||r<=0?1:r}(r.get("page")),o=r.get("status"),{whereSql:i,params:n}=function(e){let r=["alert_timestamp >= NOW() - ($1::int || ' days')::interval"],t=[e.days],a=2;return"Approved"===e.status?r.push("(claude_approved = TRUE OR claude_decision = 'APPROVE')"):"Rejected"===e.status&&r.push("(claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED')"),e.strategy&&"All"!==e.strategy&&(r.push(`strategy = $${a}`),t.push(e.strategy),a+=1),e.pair&&"All"!==e.pair&&(r.push(`(pair = $${a} OR epic ILIKE $${a+1})`),t.push(e.pair,`%${e.pair}%`),a+=2),{whereSql:`WHERE ${r.join(" AND ")}`,params:t}}({days:t,status:o,strategy:r.get("strategy"),pair:r.get("pair")});try{let[e,r,t,o,p]=await Promise.all([u.B.query("SELECT DISTINCT strategy FROM alert_history WHERE strategy IS NOT NULL ORDER BY strategy"),u.B.query("SELECT DISTINCT pair FROM alert_history WHERE pair IS NOT NULL ORDER BY pair"),u.B.query(`
        SELECT
          COUNT(*) as total_alerts,
          COUNT(CASE WHEN claude_approved = TRUE OR claude_decision = 'APPROVE' THEN 1 END) as approved,
          COUNT(CASE WHEN claude_approved = FALSE OR claude_decision = 'REJECT' OR alert_level = 'REJECTED' THEN 1 END) as rejected,
          ROUND(AVG(claude_score)::numeric, 2) as avg_score
        FROM alert_history
        ${i}
        `,n),u.B.query(`
        SELECT COUNT(*) as total
        FROM alert_history
        ${i}
        `,n),u.B.query(`
        SELECT
          id,
          alert_timestamp,
          epic,
          pair,
          signal_type,
          strategy,
          price,
          market_session,
          claude_score,
          claude_decision,
          claude_approved,
          claude_reason,
          claude_mode,
          claude_raw_response,
          vision_chart_url,
          status,
          alert_level,
          htf_candle_direction,
          htf_candle_direction_prev
        FROM alert_history
        ${i}
        ORDER BY alert_timestamp DESC
        LIMIT $${n.length+1} OFFSET $${n.length+2}
        `,[...n,a,(s-1)*a])]),d=["All",...(e.rows??[]).map(e=>e.strategy)],c=["All",...(r.rows??[]).map(e=>e.pair)],_=t.rows?.[0]??{},E=Number(o.rows?.[0]?.total??0),m=Number(_.approved??0),g=Number(_.rejected??0),y=Number(_.avg_score??0),h=(p.rows??[]).map(e=>({...e,price:null==e.price?null:Number(e.price),claude_score:null==e.claude_score?null:Number(e.claude_score),claude_approved:null==e.claude_approved?null:!!e.claude_approved}));return l.NextResponse.json({filters:{strategies:d,pairs:c},stats:{total_alerts:E,approved:m,rejected:g,avg_score:y,approval_rate:E?m/E*100:0},alerts:h,page:s,total_pages:E?Math.ceil(E/a):1,total_alerts:E})}catch(e){return console.error("Failed to load alert history",e),l.NextResponse.json({error:"Failed to load alert history"},{status:500})}}let d=new s.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/forex/alert-history/route",pathname:"/api/forex/alert-history",filename:"route",bundlePath:"app/api/forex/alert-history/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/alert-history/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:c,staticGenerationAsyncStorage:_,serverHooks:E}=d,m="/api/forex/alert-history/route";function g(){return(0,i.patchFetch)({serverHooks:E,staticGenerationAsyncStorage:_})}},1577:(e,r,t)=>{t.d(r,{B:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var r=require("../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),a=r.X(0,[5822,9967],()=>t(6142));module.exports=a})();