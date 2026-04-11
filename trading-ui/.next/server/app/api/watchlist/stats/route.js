"use strict";(()=>{var t={};t.id=3639,t.ids=[3639],t.modules={399:t=>{t.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:t=>{t.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},7064:(t,e,a)=>{a.r(e),a.d(e,{originalPathname:()=>E,patchFetch:()=>O,requestAsyncStorage:()=>m,routeModule:()=>p,serverHooks:()=>w,staticGenerationAsyncStorage:()=>h});var s={};a.r(s),a.d(s,{GET:()=>d,dynamic:()=>l});var r=a(7599),o=a(4294),n=a(4588),c=a(2921),i=a(740);let l="force-dynamic",u=["ema_50_crossover","ema_20_crossover","macd_bullish_cross"],_=["gap_up_continuation","rsi_oversold_bounce"];async function d(t){let{searchParams:e}=new URL(t.url),a=e.get("date"),s=await i.d.connect();try{let t=`
      SELECT watchlist_name, COUNT(*) as stock_count, MAX(scan_date) as last_scan
      FROM stock_watchlist_results
      WHERE watchlist_name = ANY($1)
        AND status = 'active'
      GROUP BY watchlist_name
    `,e=await s.query(t,[u]),r=a;if(!r){let t=`
        SELECT MAX(scan_date) as max_date
        FROM stock_watchlist_results
        WHERE watchlist_name = ANY($1)
      `,e=await s.query(t,[_]);r=e.rows[0]?.max_date||null}let o=[];if(r){let t=`
        SELECT watchlist_name, COUNT(*) as stock_count, MAX(scan_date) as last_scan
        FROM stock_watchlist_results
        WHERE watchlist_name = ANY($1)
          AND scan_date = $2
        GROUP BY watchlist_name
      `;o=(await s.query(t,[_,r])).rows}let n=`
      SELECT COUNT(*) as total
      FROM stock_instruments
      WHERE is_active = true
    `,i=await s.query(n),l=[...e.rows,...o],d={},p=null;return l.forEach(t=>{d[t.watchlist_name]=Number(t.stock_count||0),t.last_scan&&(!p||t.last_scan>p)&&(p=t.last_scan)}),c.NextResponse.json({counts:d,last_scan:p,total_stocks_scanned:Number(i.rows[0]?.total||0),event_date:r})}catch(t){return c.NextResponse.json({error:"Failed to load stats"},{status:500})}finally{s.release()}}let p=new r.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/watchlist/stats/route",pathname:"/api/watchlist/stats",filename:"route",bundlePath:"app/api/watchlist/stats/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/watchlist/stats/route.ts",nextConfigOutput:"standalone",userland:s}),{requestAsyncStorage:m,staticGenerationAsyncStorage:h,serverHooks:w}=p,E="/api/watchlist/stats/route";function O(){return(0,n.patchFetch)({serverHooks:w,staticGenerationAsyncStorage:h})}},740:(t,e,a)=>{a.d(e,{d:()=>r}),function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}();let s=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",r=Object(function(){var t=Error("Cannot find module 'pg'");throw t.code="MODULE_NOT_FOUND",t}())({connectionString:s,max:10})}};var e=require("../../../../webpack-runtime.js");e.C(t);var a=t=>e(e.s=t),s=e.X(0,[5822,9967],()=>a(7064));module.exports=s})();