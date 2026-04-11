"use strict";(()=>{var t={};t.id=4381,t.ids=[4381],t.modules={399:t=>{t.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:t=>{t.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},7071:(t,e,r)=>{r.r(e),r.d(e,{originalPathname:()=>m,patchFetch:()=>g,requestAsyncStorage:()=>E,routeModule:()=>d,serverHooks:()=>N,staticGenerationAsyncStorage:()=>u});var s={};r.r(s),r.d(s,{GET:()=>_,dynamic:()=>p});var o=r(7599),a=r(4294),i=r(4588),n=r(2921),l=r(1577);let p="force-dynamic";async function _(t){let{searchParams:e}=new URL(t.url),r=function(t){if(!t)return 7;let e=Number(t);return!Number.isFinite(e)||e<=0?7:e}(e.get("days")),s=new Date;s.setDate(s.getDate()-r);try{let t=(await l.B.query(`
      SELECT
        COUNT(*) as total_trades,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades,
        COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losing_trades,
        COUNT(CASE WHEN status IN ('pending', 'pending_limit') THEN 1 END) as pending_trades,
        COALESCE(SUM(profit_loss), 0) as total_profit_loss,
        COALESCE(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as avg_profit,
        COALESCE(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 0) as avg_loss,
        COALESCE(MAX(profit_loss), 0) as largest_win,
        COALESCE(MIN(profit_loss), 0) as largest_loss
      FROM trade_log
      WHERE timestamp >= $1
      `,[s])).rows[0]??{},e=Number(t.total_trades??0),r=Number(t.winning_trades??0),o=Number(t.losing_trades??0),a=Number(t.pending_trades??0),i=Number(t.total_profit_loss??0),p=Number(t.avg_profit??0),_=Number(t.avg_loss??0),d=Number(t.largest_win??0),E=Number(t.largest_loss??0),u=o>0&&_<0?p*r/Math.abs(_*o):Number.POSITIVE_INFINITY,N=(await l.B.query(`
      SELECT
        symbol,
        COUNT(*) as trades,
        COALESCE(SUM(profit_loss), 0) as total_pnl,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins
      FROM trade_log
      WHERE timestamp >= $1
      GROUP BY symbol
      ORDER BY total_pnl DESC
      `,[s])).rows??[],m=N[0]?.symbol??"None",g=N.length?N[N.length-1].symbol:"None",f=await l.B.query(`
      SELECT
        DATE(timestamp) as date,
        SUM(profit_loss) as daily_pnl,
        COUNT(*) as trade_count
      FROM trade_log
      WHERE timestamp >= $1
        AND profit_loss IS NOT NULL
      GROUP BY DATE(timestamp)
      ORDER BY date ASC
      `,[s]),C=await l.B.query(`
      SELECT
        t.id,
        t.symbol,
        t.direction,
        t.timestamp,
        t.status,
        t.profit_loss,
        t.pnl_currency,
        a.strategy
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      ORDER BY t.timestamp DESC
      LIMIT 12
      `,[s]);return n.NextResponse.json({stats:{total_trades:e,winning_trades:r,losing_trades:o,pending_trades:a,total_profit_loss:i,win_rate:e>0?r/e*100:0,avg_profit:p,avg_loss:_,profit_factor:u,largest_win:d,largest_loss:E,best_pair:m,worst_pair:g,active_pairs:N.map(t=>t.symbol)},daily_pnl:f.rows??[],recent_trades:C.rows??[]})}catch(t){return console.error("Failed to load forex overview",t),n.NextResponse.json({error:"Failed to load forex overview"},{status:500})}}let d=new o.AppRouteRouteModule({definition:{kind:a.x.APP_ROUTE,page:"/api/forex/overview/route",pathname:"/api/forex/overview",filename:"route",bundlePath:"app/api/forex/overview/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/forex/overview/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:E,staticGenerationAsyncStorage:u,serverHooks:N}=d,m="/api/forex/overview/route";function g(){return(0,i.patchFetch)({serverHooks:N,staticGenerationAsyncStorage:u})}},1577:(t,e,r)=>{r.d(e,{B:()=>o}),function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}();let s=process.env.DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/forex",o=Object(function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}())({connectionString:s,max:10})}};var e=require("../../../../webpack-runtime.js");e.C(t);var r=t=>e(e.s=t),s=e.X(0,[5822,9967],()=>r(7071));module.exports=s})();