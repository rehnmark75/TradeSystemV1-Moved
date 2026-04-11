"use strict";(()=>{var e={};e.id=2822,e.ids=[2822],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},3075:(e,t,r)=>{r.r(t),r.d(t,{originalPathname:()=>w,patchFetch:()=>h,requestAsyncStorage:()=>g,routeModule:()=>p,serverHooks:()=>m,staticGenerationAsyncStorage:()=>E});var o={};r.r(o),r.d(o,{GET:()=>u,dynamic:()=>c});var a=r(7599),n=r(4294),s=r(4588),i=r(2921),l=r(740);let c="force-dynamic",_=e=>{if(null==e)return null;let t=Number(e);return Number.isNaN(t)?null:t},d=e=>e?new Date(e).toISOString().slice(0,10):"unknown";async function u(e){let{searchParams:t}=new URL(e.url),r=Number(t.get("days")||30),o=Number(t.get("trendDays")||7),a=await l.d.connect();try{let e=`
      SELECT total_value, invested, available, recorded_at
      FROM broker_account_balance
      ORDER BY recorded_at DESC
      LIMIT 1
    `,t=(await a.query(e)).rows[0]||null,n=`
      SELECT recorded_at, total_value
      FROM broker_account_balance
      WHERE recorded_at >= NOW() - INTERVAL '${o} days'
      ORDER BY recorded_at ASC
    `,s=(await a.query(n)).rows||[],l=s[0],c=s[s.length-1],u=l&&c?_(c.total_value)-_(l.total_value):0,p=l&&_(l.total_value)?u/_(l.total_value)*100:0,g=`
      SELECT completed_at
      FROM broker_sync_log
      WHERE status = 'completed'
      ORDER BY completed_at DESC
      LIMIT 1
    `,E=await a.query(g),m=E.rows[0]?.completed_at||null,w=`
      SELECT
        deal_id,
        ticker,
        side,
        quantity,
        open_price,
        close_price,
        profit,
        profit_pct,
        duration_hours,
        open_time,
        close_time
      FROM broker_trades
      WHERE status = 'closed'
        AND close_time >= NOW() - INTERVAL '${r} days'
      ORDER BY close_time DESC
    `,h=(await a.query(w)).rows||[],v=`
      SELECT
        deal_id,
        ticker,
        side,
        quantity,
        open_price,
        current_price,
        profit,
        stop_loss,
        take_profit,
        open_time
      FROM broker_trades
      WHERE status = 'open'
      ORDER BY open_time DESC
    `,R=(await a.query(v)).rows||[],f=0,b=0,y=0,O=0,k=0,D=0,S=0,T=0,N=0,C=0,x=0,L=0,q=0,A=0,M=0,F=[],j=[],I=[],W=[],B=[...h].sort((e,t)=>new Date(e.close_time).getTime()-new Date(t.close_time).getTime()),P=0,U=0,H=0,Y=0,V={},$={};h.forEach(e=>{let t=_(e.profit)||0,r=_(e.profit_pct),o=_(e.duration_hours),a=e.side,n=d(e.close_time);y+=t,t>=0?(f+=t,O+=1,I.push(t),null!==r&&F.push(r),t>D&&(D=t)):(b+=t,k+=1,W.push(t),null!==r&&j.push(r),t<S&&(S=t)),null!==o&&(T+=o,N+=1),"long"===a&&(C+=1,A+=t,t>0&&(L+=1)),"short"===a&&(x+=1,M+=t,t>0&&(q+=1)),V[n]||(V[n]={pnl:0,count:0}),V[n].pnl+=t,V[n].count+=1,$[e.ticker]||($[e.ticker]={trades:0,wins:0,pnl:0}),$[e.ticker].trades+=1,$[e.ticker].pnl+=t,t>0&&($[e.ticker].wins+=1)}),B.forEach(e=>{(_(e.profit)||0)>=0?(P+=1,U=0):(U+=1,P=0),P>H&&(H=P),U>Y&&(Y=U)});let G=h.length,z=G?O/G*100:0,K=I.length?I.reduce((e,t)=>e+t,0)/I.length:0,X=W.length?W.reduce((e,t)=>e+t,0)/W.length:0,J=F.length?F.reduce((e,t)=>e+t,0)/F.length:0,Q=j.length?j.reduce((e,t)=>e+t,0)/j.length:0,Z=0!==b?f/Math.abs(b):0,ee=G?y/G:0,et=N?T/N:0,er=C?L/C*100:0,eo=x?q/x*100:0,ea=`
      SELECT recorded_at, total_value
      FROM broker_account_balance
      WHERE recorded_at >= NOW() - INTERVAL '${r} days'
      ORDER BY recorded_at ASC
    `,en=(await a.query(ea)).rows||[],es=0,ei=0;en.forEach(e=>{let t=_(e.total_value)||0;t>es&&(es=t);let r=es?es-t:0;r>ei&&(ei=r)});let el=es?ei/es*100:0;return i.NextResponse.json({balance:t,trend:{change:u,change_pct:p,trend:u>0?"up":u<0?"down":"neutral",data_points:s.length},last_sync:m,stats:{total_trades:G,winning_trades:O,losing_trades:k,win_rate:z,total_profit:f,total_loss:b,net_profit:y,avg_profit:K,avg_loss:X,avg_profit_pct:J,avg_loss_pct:Q,largest_win:D,largest_loss:S,profit_factor:Z,expectancy:ee,max_drawdown:ei,max_drawdown_pct:el,max_consecutive_wins:H,max_consecutive_losses:Y,avg_trade_duration_hours:et,long_trades:C,short_trades:x,long_win_rate:er,short_win_rate:eo,long_profit:A,short_profit:M},open_positions:R.map(e=>{let t=_(e.open_price)||0,r=_(e.current_price)||0,o=0;return t&&r&&(o="long"===e.side?(r-t)/t*100:(t-r)/t*100),{...e,entry_price:t,current_price:r,unrealized_pnl:_(e.profit)||0,profit_pct:o}}),closed_trades:h.slice(0,50),by_day:Object.entries(V).map(([e,t])=>({date:e,pnl:t.pnl,count:t.count})),by_ticker:Object.entries($).map(([e,t])=>({ticker:e,trades:t.trades,win_rate:t.trades?t.wins/t.trades*100:0,pnl:t.pnl})),equity_curve:en})}catch(e){return i.NextResponse.json({error:"Failed to load broker stats"},{status:500})}finally{a.release()}}let p=new a.AppRouteRouteModule({definition:{kind:n.x.APP_ROUTE,page:"/api/broker/overview/route",pathname:"/api/broker/overview",filename:"route",bundlePath:"app/api/broker/overview/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/broker/overview/route.ts",nextConfigOutput:"standalone",userland:o}),{requestAsyncStorage:g,staticGenerationAsyncStorage:E,serverHooks:m}=p,w="/api/broker/overview/route";function h(){return(0,s.patchFetch)({serverHooks:m,staticGenerationAsyncStorage:E})}},740:(e,t,r)=>{r.d(t,{d:()=>a}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let o=process.env.STOCKS_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/stocks",a=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:o,max:10})}};var t=require("../../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),o=t.X(0,[5822,9967],()=>r(3075));module.exports=o})();