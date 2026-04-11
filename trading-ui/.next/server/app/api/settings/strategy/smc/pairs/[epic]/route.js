"use strict";(()=>{var e={};e.id=1843,e.ids=[1843],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},9933:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>m,patchFetch:()=>w,requestAsyncStorage:()=>g,routeModule:()=>y,serverHooks:()=>R,staticGenerationAsyncStorage:()=>E});var a={};t.r(a),t.d(a,{DELETE:()=>f,GET:()=>l,PUT:()=>_,dynamic:()=>d});var s=t(7599),i=t(4294),o=t(4588),n=t(2921),u=t(4372);let d="force-dynamic";async function p(e){let r=e??u.A,t=await r.query(`
      SELECT id
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `);return t.rows[0]?.id??null}async function c(e){let r=e??u.A;return(await r.query(`
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'smc_simple_pair_overrides'
    `)).rows.map(e=>e.column_name)}async function l(e,{params:r}){try{let e=await p();if(!e)return n.NextResponse.json({error:"No active SMC config found"},{status:404});let t=await u.A.query(`
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,[e,r.epic]);if(!t.rows[0])return n.NextResponse.json({error:"Pair override not found"},{status:404});return n.NextResponse.json(t.rows[0])}catch(e){return console.error("Failed to load pair override",e),n.NextResponse.json({error:"Failed to load pair override"},{status:500})}}async function _(e,{params:r}){let t=await e.json().catch(()=>null);if(!t||"object"!=typeof t)return n.NextResponse.json({error:"Invalid request body"},{status:400});let{updates:a,updated_by:s,change_reason:i,updated_at:o}=t;if(!a||"object"!=typeof a)return n.NextResponse.json({error:"Missing updates payload"},{status:400});if(!s||!i)return n.NextResponse.json({error:"updated_by and change_reason are required"},{status:400});if(!o)return n.NextResponse.json({error:"updated_at is required for optimistic locking"},{status:400});let d=await u.A.connect();try{await d.query("BEGIN");let e=await p(d);if(!e)return await d.query("ROLLBACK"),n.NextResponse.json({error:"No active SMC config found"},{status:404});let t=(await d.query(`
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,[e,r.epic])).rows[0];if(!t)return await d.query("ROLLBACK"),n.NextResponse.json({error:"Pair override not found"},{status:404});if(function(e){if(!e)return"";let r=e instanceof Date?e:new Date(String(e));return Number.isNaN(r.getTime())?"":r.toISOString()}(t.updated_at)!==o)return await d.query("ROLLBACK"),n.NextResponse.json({error:"conflict",message:"Pair override was updated by another user",current_updated_at:t.updated_at,updated_by:t.updated_by,current_override:t},{status:409});let u=await c(d),l=new Set(u.filter(e=>!["id","config_id","created_at","updated_at","updated_by","change_reason","epic"].includes(e))),_=Object.keys(a).filter(e=>l.has(e));if(0===_.length)return await d.query("ROLLBACK"),n.NextResponse.json({error:"No valid fields to update"},{status:400});let f=[],y=[];_.forEach((e,r)=>{f.push(`${e} = $${r+1}`),y.push(a[e])}),f.push(`updated_by = $${_.length+1}`),f.push(`change_reason = $${_.length+2}`),y.push(s),y.push(i),y.push(t.id),y.push(o);let g=`
      UPDATE smc_simple_pair_overrides
      SET ${f.join(", ")}
      WHERE id = $${_.length+3} AND updated_at = $${_.length+4}
      RETURNING *
    `,E=await d.query(g,y);if(0===E.rowCount)return await d.query("ROLLBACK"),n.NextResponse.json({error:"conflict",message:"Pair override was updated by another user",current_updated_at:t.updated_at,updated_by:t.updated_by,current_override:t},{status:409});let R={};return _.forEach(e=>{R[e]=t[e]}),await d.query(`
        INSERT INTO smc_simple_config_audit
          (config_id, pair_override_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, $2, 'PAIR_UPDATE', $3, $4, $5, $6)
      `,[e,E.rows[0]?.id??null,s,i,JSON.stringify(R),JSON.stringify(a)]),await d.query("COMMIT"),n.NextResponse.json(E.rows[0])}catch(e){return await d.query("ROLLBACK"),console.error("Failed to update pair override",e),n.NextResponse.json({error:"Failed to update pair override"},{status:500})}finally{d.release()}}async function f(e,{params:r}){let t=await e.json().catch(()=>null),a=t?.updated_by,s=t?.change_reason;if(!a||!s)return n.NextResponse.json({error:"updated_by and change_reason are required"},{status:400});let i=await u.A.connect();try{await i.query("BEGIN");let e=await p(i);if(!e)return await i.query("ROLLBACK"),n.NextResponse.json({error:"No active SMC config found"},{status:404});let t=(await i.query(`
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,[e,r.epic])).rows[0];if(!t)return await i.query("ROLLBACK"),n.NextResponse.json({error:"Pair override not found"},{status:404});return await i.query(`
        DELETE FROM smc_simple_pair_overrides
        WHERE id = $1
      `,[t.id]),await i.query(`
        INSERT INTO smc_simple_config_audit
          (config_id, pair_override_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, $2, 'PAIR_DELETE', $3, $4, $5, $6)
      `,[e,t.id,a,s,JSON.stringify(t),null]),await i.query("COMMIT"),n.NextResponse.json({success:!0})}catch(e){return await i.query("ROLLBACK"),console.error("Failed to delete pair override",e),n.NextResponse.json({error:"Failed to delete pair override"},{status:500})}finally{i.release()}}let y=new s.AppRouteRouteModule({definition:{kind:i.x.APP_ROUTE,page:"/api/settings/strategy/smc/pairs/[epic]/route",pathname:"/api/settings/strategy/smc/pairs/[epic]",filename:"route",bundlePath:"app/api/settings/strategy/smc/pairs/[epic]/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/settings/strategy/smc/pairs/[epic]/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:g,staticGenerationAsyncStorage:E,serverHooks:R}=y,m="/api/settings/strategy/smc/pairs/[epic]/route";function w(){return(0,o.patchFetch)({serverHooks:R,staticGenerationAsyncStorage:E})}},4372:(e,r,t)=>{t.d(r,{A:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.STRATEGY_CONFIG_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/strategy_config",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var r=require("../../../../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),a=r.X(0,[5822,9967],()=>t(9933));module.exports=a})();