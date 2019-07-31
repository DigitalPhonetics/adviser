import React, {Component} from 'react';
import TurnInfoItem from './turninfoitem';

class TurnInfoList extends Component {

    constructor(props) {
        super(props);
    }
 
    render() {
        console.log("Turn Info");
        console.log(this.props.turnInfo);
        
        var turnInfo;
        if(Array.isArray(this.props.turnInfo)) {
            turnInfo = this.props.turnInfo;
        }
        else {
            turnInfo = [];
        } 
        
        return <div className="TurnInfoList">
        {
            turnInfo.map(info => (
                <TurnInfoItem index={info.index} name={info.name} diff={info.diff} count={turnInfo.length}/>
            ))
        }
        <style jsx>{`
        .TurnInfoList {
        display: grid;
        grid-template-columns: 100%;
        height: 100vh;
        max-height: 100vh;
        overflow-y: auto;
        }
        `}</style>
     </div>
    }
}

export default TurnInfoList;