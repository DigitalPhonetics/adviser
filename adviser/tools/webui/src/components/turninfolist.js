import React, {Component} from 'react';
import TurnInfoItem from './turninfoitem';

// import '../App.css';

class TurnInfoList extends Component {

    // constructor(props) {
    //     super(props);
    // }
 
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
                <TurnInfoItem key={info.index} index={info.index} name={info.name} diff={info.diff} count={turnInfo.length}/>
            ))
        }
     </div>
    }
}

export default TurnInfoList;