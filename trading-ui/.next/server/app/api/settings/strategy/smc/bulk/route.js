"use strict";(()=>{var e={};e.id=9250,e.ids=[9250],e.modules={399:e=>{e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},625:(e,r,t)=>{t.r(r),t.d(r,{originalPathname:()=>f,patchFetch:()=>R,requestAsyncStorage:()=>m,routeModule:()=>g,serverHooks:()=>E,staticGenerationAsyncStorage:()=>y});var a={};t.r(a),t.d(a,{POST:()=>_,dynamic:()=>c});var s=t(7599),i=t(4294),n=t(4588),o=t(2921),u=t(4372);let c="force-dynamic";async function p(e){let r=e??u.A;return(await r.query(`
      SELECT *
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `)).rows[0]??null}async function d(e,r,t){return(await e.query(`
      SELECT *
      FROM smc_simple_pair_overrides
      WHERE config_id = $1 AND epic = $2
      LIMIT 1
    `,[r,t])).rows[0]??null}async function l(e){return(await e.query(`
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'smc_simple_pair_overrides'
    `)).rows.map(e=>e.column_name)}async function _(e){let r=await e.json().catch(()=>null);if(!r||"object"!=typeof r)return o.NextResponse.json({error:"Invalid request body"},{status:400});let{action:t,epics:a,source_epic:s,updated_by:i,change_reason:n}=r;if(!t||!Array.isArray(a)||0===a.length)return o.NextResponse.json({error:"action and epics are required"},{status:400});if(!i||!n)return o.NextResponse.json({error:"updated_by and change_reason are required"},{status:400});let c=await u.A.connect();try{await c.query("BEGIN");let e=await p(c);if(!e)return await c.query("ROLLBACK"),o.NextResponse.json({error:"No active SMC config found"},{status:404});let r=[];if("reset"===t)await c.query(`
          DELETE FROM smc_simple_pair_overrides
          WHERE config_id = $1 AND epic = ANY($2)
        `,[e.id,a]),r.push(...a);else if("copy-global"===t)await c.query(`
          DELETE FROM smc_simple_pair_overrides
          WHERE config_id = $1 AND epic = ANY($2)
        `,[e.id,a]),r.push(...a);else{if("copy-pair"!==t)return await c.query("ROLLBACK"),o.NextResponse.json({error:"Unknown action"},{status:400});if(!s)return await c.query("ROLLBACK"),o.NextResponse.json({error:"source_epic is required for copy-pair"},{status:400});let u=await d(c,e.id,s);if(!u)return await c.query("ROLLBACK"),o.NextResponse.json({error:"Source override not found"},{status:404});let p=(await l(c)).filter(e=>!["id","config_id","epic","created_at","updated_at","updated_by","change_reason"].includes(e)),_={};for(let r of(p.forEach(e=>{_[e]=u[e]}),a)){let t=["config_id","epic",...p,"updated_by","change_reason"],a=[e.id,r,...p.map(e=>_[e]),i,n],s=t.map((e,r)=>`$${r+1}`).join(", "),o=p.map(e=>`${e} = EXCLUDED.${e}`).concat(["updated_by = EXCLUDED.updated_by","change_reason = EXCLUDED.change_reason"]).join(", ");await c.query(`
            INSERT INTO smc_simple_pair_overrides (${t.join(", ")})
            VALUES (${s})
            ON CONFLICT (config_id, epic)
            DO UPDATE SET ${o}
          `,a)}r.push(...a)}return await c.query(`
        INSERT INTO smc_simple_config_audit
          (config_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, 'BULK_UPDATE', $2, $3, $4, $5)
      `,[e.id,i,n,null,JSON.stringify({action:t,epics:r,source_epic:s})]),await c.query("COMMIT"),o.NextResponse.json({success:!0,action:t,affected:r})}catch(e){return await c.query("ROLLBACK"),console.error("Failed to apply bulk override action",e),o.NextResponse.json({error:"Failed to apply bulk override action"},{status:500})}finally{c.release()}}let g=new s.AppRouteRouteModule({definition:{kind:i.x.APP_ROUTE,page:"/api/settings/strategy/smc/bulk/route",pathname:"/api/settings/strategy/smc/bulk",filename:"route",bundlePath:"app/api/settings/strategy/smc/bulk/route"},resolvedPagePath:"/home/hr/Projects/TradeSystemV1/trading-ui/app/api/settings/strategy/smc/bulk/route.ts",nextConfigOutput:"standalone",userland:a}),{requestAsyncStorage:m,staticGenerationAsyncStorage:y,serverHooks:E}=g,f="/api/settings/strategy/smc/bulk/route";function R(){return(0,n.patchFetch)({serverHooks:E,staticGenerationAsyncStorage:y})}},4372:(e,r,t)=>{t.d(r,{A:()=>s}),function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}();let a=process.env.STRATEGY_CONFIG_DATABASE_URL||"postgresql://postgres:postgres@postgres:5432/strategy_config",s=Object(function(){var e=Error("Cannot find module 'pg'");throw e.code="MODULE_NOT_FOUND",e}())({connectionString:a,max:10})}};var r=require("../../../../../../webpack-runtime.js");r.C(e);var t=e=>r(r.s=e),a=r.X(0,[5822,9967],()=>t(625));module.exports=a})();